'''
train.py

* Training and validation functions
* wandb logging
'''

import time
import gc
import os
import numpy as np
import wandb
import torch
import torch.nn as nn
import torch.optim as optim
from src.losses import ssim, SNR, PSNR, StegoLoss, calc_ber, signal_noise_ratio, batch_signal_noise_ratio, \
    batch_calc_ber
from src.visualization import viz2paper, viz4seq
from src.umodel import StegoUNet
import re


def save_checkpoint(state, is_best, filename=os.path.join(os.environ.get('OUT_PATH'), 'checkpoint.pt')):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print("=> Saving a new best model")
        print(f'SAVING TO: {filename}')
        torch.save(state, filename)  # save checkpoint
    else:
        print("=> Loss did not improve")


def train(model, tr_loader, vd_loader, beta, lam, alpha, gamma, lr, epochs=5, val_itvl=500, val_size=50,
          prev_epoch=None, prev_i=None,
          summary=None, slide=1, experiment=0, transform='cosine', stft_small=True, ft_container='mag', thet=0,
          dtw=False):
    # Initialize wandb logs
    wandb.init(project='WavmarkReproduce')
    if summary is not None:
        wandb.run.name = summary
        wandb.run.save()
    wandb.watch(model)

    # Prepare to device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Using device: {device}')

    # Parallelize on GPU
    if torch.cuda.device_count() > 1:
        print(f"Let's use {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)

    model.to(device)

    # Set to training mode
    model.train()

    # Number of parameters in model
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Number of model parameters: {num_params}')

    # Set optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)
    # optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    # optimizer = optim.Adam([{"params": model.watermark_fc.parameters(), "lr": lr / 1e1},
    #                         {"params": model.hinet.parameters(), "lr": lr / 1e1},
    #                         {"params": model.hinet_r.parameters(), "lr": lr},
    #                         {"params": model.watermark_fc_back.parameters(), "lr": lr}],
    #                        lr=lr)

    # Initialize waveform loss constructor
    criterion_audio = nn.MSELoss()
    criterion_watermark = nn.MSELoss()
    criterion_restore_audio = nn.MSELoss()
    criterion_contrast = nn.TripletMarginLoss()
    criterion_audio_name = criterion_audio.__class__.__name__[
                           :re.search("(?=Loss)", criterion_audio.__class__.__name__).span()[0]]
    criterion_wm_name = criterion_watermark.__class__.__name__[
                        :re.search("(?=Loss)", criterion_watermark.__class__.__name__).span()[0]]

    # load watermark model as attacker
    # change initial attacker, updating freq
    wm_model = None
    if model.transform == "WM":
        wm_len = 8
        ckpt_path = f"1-multi_IDwl{wm_len}lr1e-4audioMSElam100/50-1-multi_IDwl{wm_len}lr1e-4audioMSElam100.pt"
        # ckpt_path = f"1-multi_WMwl{wm_len}lr1e-4audioMSElam100/23-1-multi_WMwl{wm_len}lr1e-4audioMSElam100.pt"
        state_dict = torch.load(os.path.join(os.environ.get('OUT_PATH'), ckpt_path))["state_dict"]
        if model.share_param:
            state_dict = {k:v for k, v in state_dict.items() if "hinet_r" not in k}
        wm_model = StegoUNet("ID", model.num_points, model.n_fft, model.hop_length, False, model.num_layers,
                             wm_len, 0, model.share_param)
        wm_model.load_state_dict(state_dict, strict=False)
        wm_model.to(device)

        last_epoch_state = wm_model.state_dict()

    # Initialize best val
    best_loss = np.inf
    best_snr = - np.inf
    ber_threshold = 1 / 32 / 2

    # Start training ...
    ini = time.time()
    for epoch in range(epochs):

        # if prev_epoch != None and epoch < prev_epoch - 1: continue  # Checkpoint pass

        # Initialize training metrics storage
        train_loss, train_audio_loss, train_watermark_loss, train_restore_audio_loss, train_contrast_loss, train_snr, train_ber = [], [], [], [], [], [], []
        vd_loss, vd_audio_loss, vd_watermark_loss, vd_restore_audio_loss, vd_contrast_loss, vd_snr, vd_ber = [], [], [], [], [], [], []

        # Print headers for the losses
        print()
        print(f' Iter.     Time  '
              f'Tr. Loss '
              f' Au. {criterion_audio_name:3} '
              f'rAu. MSE '
              f'rAu. Tri '
              f' Wm. {criterion_wm_name:3} '
              f' Au. SNR '
              f' Wm. BER ')
        for i, data in enumerate(tr_loader):

            # if prev_i != None and i < prev_i - 1: continue  # Checkpoint pass

            # Load data from the loader
            # (B,secret_len), (B,secret_len,), (B,L)  B=1
            secrets, secrets_bin, covers = data[0][0].to(device), data[0][1].to(device), data[1].to(device)
            secrets = secrets.type(torch.cuda.FloatTensor)
            transcripts, text_prompts = data[2], data[3]
            if data[4]:
                shift_sound = [data[4][0].to(device), data[4][1].to(device)]
            else:
                shift_sound = data[4]

            optimizer.zero_grad()

            # Forward through the model
            # (B,N,T,C), (B,N,T,C), (B,L), (B,secret_len)
            cover_fft, containers_fft, container_wav, revealed, audio_revealed = model(secrets,
                                                                                       covers,
                                                                                       transcripts,
                                                                                       text_prompts,
                                                                                       shift_sound=shift_sound,
                                                                                       wm_model=wm_model)

            # loss
            # loss, loss_cover, loss_secret, loss_spectrum = StegoLoss(secrets, cover_fft, containers_fft, None,
            #                                                          revealed, beta)
            loss_watermark = criterion_watermark(revealed, secrets)
            loss_audio = criterion_audio(covers, container_wav)
            loss_restore_audio = criterion_restore_audio(covers, audio_revealed)
            loss_contrast = criterion_contrast(audio_revealed, covers, container_wav)
            loss_total = beta * loss_watermark + lam * loss_audio + alpha * loss_restore_audio + gamma * loss_contrast

            with torch.autograd.set_detect_anomaly(True):
                loss_total.backward()
            optimizer.step()

            # Compute audio metrics
            snr_audio = batch_signal_noise_ratio(covers, container_wav)
            ber = batch_calc_ber(revealed, secrets)

            # Append and average the new losses
            train_loss.append(loss_total.detach().item())
            train_audio_loss.append(loss_audio.detach().item())
            train_restore_audio_loss.append(loss_restore_audio.detach().item())
            train_contrast_loss.append(loss_contrast.detach().item())
            train_watermark_loss.append(loss_watermark.detach().item())
            train_snr.append(snr_audio)
            train_ber.append(ber.item())

            print(f"(#{i:4})[{np.round(time.time() - ini):5} s] "
                  f"{train_loss[-1]:8.4f} "
                  f"{train_audio_loss[-1]:8.4f} "
                  f"{train_restore_audio_loss[-1]:8.4f} "
                  f"{train_contrast_loss[-1]:8.4f} "
                  f"{train_watermark_loss[-1]:8.4f} "
                  f"{train_snr[-1]:8.4f} "
                  f"{train_ber[-1]:8.4f}")

            # Log training statistics to wandb
            wandb.log({
                'iter_tr_loss': train_loss[-1],
                'iter_tr_audio_loss': train_audio_loss[-1],
                'iter_tr_restore_audio_loss': train_restore_audio_loss[-1],
                'iter_tr_contrast_loss': train_contrast_loss[-1],
                'iter_tr_watermark_loss': train_watermark_loss[-1],
                'iter_tr_SNR': train_snr[-1],
                'iter_tr_BER': train_ber[-1],
                'iter_tr_lr': optimizer.param_groups[0]['lr']
            })

            # Every 'val_itvl' iterations or at the end of epoch, do a validation step
            if (i % val_itvl == 0) and (i != 0) or i == len(tr_loader) - 1:
                valid_loss, valid_audio_loss, valid_watermark_loss, valid_snr, valid_ber, valid_rst_loss, valid_cts_loss = validate(
                    model, vd_loader, beta=beta, lmd=lam, alpha=alpha, gamma=gamma, val_size=val_size,
                    audio_criterion=criterion_audio, wm_criterion=criterion_watermark,
                    rst_audio_criterion=criterion_restore_audio, contrast_criterion=criterion_contrast,
                    tr_i=i, epoch=epoch, wm_model=wm_model)

                vd_loss.append(valid_loss)
                vd_audio_loss.append(valid_audio_loss)
                vd_restore_audio_loss.append(valid_rst_loss)
                vd_contrast_loss.append(valid_cts_loss)
                vd_watermark_loss.append(valid_watermark_loss)
                vd_snr.append(valid_snr)
                vd_ber.append(valid_ber)

                if ber_threshold < valid_ber:
                    is_best = bool(valid_loss < best_loss)
                else:
                    is_best = bool(valid_snr > best_snr)
                # TODO: print best loss and best snr, record parameters' info like current learning rate
                print(f"Current best -> best loss:{best_loss}, best snr: {best_snr}")
                best_loss = min(best_loss, valid_loss)
                best_snr = max(best_snr, valid_snr)

                # Save checkpoint if is a new best
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'beta': beta,
                    'lr': lr,
                    'i': i + 1,
                    'tr_loss': train_loss,
                    'tr_audio_loss': train_audio_loss,
                    'tr_restore_audio_loss': train_restore_audio_loss,
                    'tr_contrast_loss': train_contrast_loss,
                    'tr_watermark_loss': train_watermark_loss,
                    'tr_snr': train_snr,
                    'tr_ber': train_ber,
                    'vd_loss': vd_loss,
                    'vd_audio_loss': vd_audio_loss,
                    'vd_restore_audio_loss': vd_restore_audio_loss,
                    'vd_contrast_loss': vd_contrast_loss,
                    'vd_watermark_loss': vd_watermark_loss,
                    'vd_snr': vd_snr,
                    'vd_ber': vd_ber,
                    'audio_loss_name': criterion_audio_name,
                    'wm_loss_name': criterion_wm_name
                }, is_best=is_best, filename=os.path.join(os.environ.get('OUT_PATH'),
                                                          f'{experiment}-{summary}/{epoch + 1}-{experiment}-{summary}.pt'))
                print(f"Current best -> best loss:{best_loss}, best snr: {best_snr}")

                # Print headers again to resume training
                print()
                print(f' Iter.     Time  '
                      f'Tr. Loss '
                      f' Au. {criterion_audio_name:3} '
                      f'rAu. MSE '
                      f'rAu. Tri '
                      f' Wm. {criterion_wm_name:3} '
                      f' Au. SNR '
                      f' Wm. BER ')

        if model.transform == "WM":
            # update attacker
            last_epoch_state = model.state_dict()
            if (epoch + 1) > 26 and (epoch + 1) % 2 == 1 or 27 > (epoch + 1) > 4 and (epoch + 1) % 5 == 0:
                print(f"Updating watermarking model attacker")
                wm_model.load_state_dict(last_epoch_state, strict=False)
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': last_epoch_state,
                    'best_loss': best_loss,
                    'beta': beta,
                    'lr': lr,
                    'i': -1,
                    'tr_loss': train_loss,
                    'tr_audio_loss': train_audio_loss,
                    'tr_restore_audio_loss': train_restore_audio_loss,
                    'tr_contrast_loss': train_contrast_loss,
                    'tr_watermark_loss': train_watermark_loss,
                    'tr_snr': train_snr,
                    'tr_ber': train_ber,
                    'vd_loss': vd_loss,
                    'vd_audio_loss': vd_audio_loss,
                    'vd_restore_audio_loss': vd_restore_audio_loss,
                    'vd_contrast_loss': vd_contrast_loss,
                    'vd_watermark_loss': vd_watermark_loss,
                    'vd_snr': vd_snr,
                    'vd_ber': vd_ber,
                    'audio_loss_name': criterion_audio_name,
                    'wm_loss_name': criterion_wm_name
                }, is_best=True, filename=os.path.join(os.environ.get('OUT_PATH'),
                                                       f'{experiment}-{summary}/{epoch + 1}-{experiment}-self-{summary}.pt'))

        # Print average training results after every epoch
        train_loss_avg = np.mean(train_loss)
        train_audio_loss_avg = np.mean(train_audio_loss)
        train_restore_audio_loss_avg = np.mean(train_restore_audio_loss)
        train_contrast_loss_avg = np.mean(train_contrast_loss)
        train_watermark_loss_avg = np.mean(train_watermark_loss)
        train_snr_avg = np.mean(train_snr)
        train_ber_avg = np.mean(train_ber)

        print()
        print(f'Epoch average:      '
              f'Tr. Loss '
              f' Au. {criterion_audio_name:3} '
              f'rAu. MSE '
              f'rAu. Tri '
              f' Wm. {criterion_wm_name:3} '
              f' Au. SNR '
              f' Wm. BER ')
        print(f'Epoch {epoch:2}/{epochs:2}:        '
              f"{train_loss_avg:8.4f} "
              f"{train_audio_loss_avg:8.4f} "
              f"{train_restore_audio_loss_avg:8.4f} "
              f"{train_contrast_loss_avg:8.4f} "
              f"{train_watermark_loss_avg:8.4f} "
              f"{train_snr_avg:8.4f} "
              f"{train_ber_avg:8.4f}")
        print()

        # Log train average loss to wandb
        wandb.log({
            'epoch_tr_loss': train_loss_avg,
            'epoch_tr_audio_loss': train_audio_loss_avg,
            'epoch_tr_restore_audio_loss': train_restore_audio_loss_avg,
            'epoch_tr_contrast_loss': train_contrast_loss_avg,
            'epoch_tr_watermark_loss': train_watermark_loss_avg,
            'epoch_tr_SNR': train_snr_avg,
            'epoch_tr_BER': train_ber_avg,
        })

    print(f"Training took {time.time() - ini} seconds")
    torch.save(model.state_dict(),
               os.path.join(os.environ.get('OUT_PATH'), f'{experiment}-{summary}/final_run_{experiment}.pt'))
    return model, train_loss_avg


def validate(model, vd_loader, beta, lmd, alpha, gamma, val_size,
             audio_criterion, wm_criterion, rst_audio_criterion, contrast_criterion, epoch=None, tr_i=None,
             **kwargs):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parallelize on GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).cuda()

    model.to(device)

    # Set to evaluation mode
    model.eval()

    valid_loss, valid_audio_loss, valid_rst_audio_loss, valid_contrast_loss, valid_watermark_loss, valid_snr, valid_ber = [], [], [], [], [], [], []

    # Start validating ...
    with torch.no_grad():
        for i, data in enumerate(vd_loader):

            # Load data from the loader
            # (B,secret_len), (B,secret_len,), (B,L)  B=1
            secrets, secrets_bin, covers = data[0][0].to(device), data[0][1].to(device), data[1].to(device)
            secrets = secrets.type(torch.cuda.FloatTensor)
            transcripts, text_prompts = data[2], data[3]
            if data[4]:
                shift_sound = [data[4][0].to(device), data[4][1].to(device)]
            else:
                shift_sound = data[4]

            # Forward through the model
            # (B,N,T,2), (B,N,T,2), (B,L), (B,secret_len)
            cover_fft, containers_fft, container_wav, revealed, audio_revealed = model(secrets,
                                                                                       covers,
                                                                                       transcripts,
                                                                                       text_prompts,
                                                                                       shift_sound=shift_sound,
                                                                                       wm_model=kwargs["wm_model"])

            # Visualize results
            if i == 0:
                fig = viz4seq(secrets[0].cpu(), revealed[0].cpu(), cover_fft.cpu(), containers_fft.cpu(), None, None)
                wandb.log({f"Revelation at epoch {epoch}, vd iteration {tr_i}": fig})

            # Compute the loss
            loss_watermark = wm_criterion(revealed, secrets)
            loss_audio = audio_criterion(covers, container_wav)
            loss_rst_audio = rst_audio_criterion(covers, audio_revealed)
            loss_contrast = contrast_criterion(audio_revealed, covers, container_wav)
            loss_total = beta * loss_watermark + lmd * loss_audio + alpha * loss_rst_audio + gamma * loss_contrast

            # Compute audio metrics
            vd_snr_audio = batch_signal_noise_ratio(covers, container_wav)
            ber = batch_calc_ber(revealed, secrets)

            valid_loss.append(loss_total.detach().item())
            valid_audio_loss.append(loss_audio.detach().item())
            valid_rst_audio_loss.append(loss_rst_audio.detach().item())
            valid_contrast_loss.append(loss_contrast.detach().item())
            valid_watermark_loss.append(loss_watermark.detach().item())
            valid_snr.append(vd_snr_audio)
            valid_ber.append(ber.item())

            # Stop validation after val_size steps
            if i >= val_size: break

        avg_valid_loss = np.mean(valid_loss)
        avg_valid_audio_loss = np.mean(valid_audio_loss)
        avg_valid_rst_audio_loss = np.mean(valid_rst_audio_loss)
        avg_valid_contrast_loss = np.mean(valid_contrast_loss)
        avg_valid_watermark_loss = np.mean(valid_watermark_loss)
        avg_valid_snr = np.mean(valid_snr)
        avg_valid_ber = np.mean(valid_ber)

        wandb.log({
            'vd_loss': avg_valid_loss,
            'vd_audio_loss': avg_valid_audio_loss,
            'vd_restore_audio_loss': avg_valid_rst_audio_loss,
            'vd_contrast_loss': avg_valid_contrast_loss,
            'vd_watermark_loss': avg_valid_watermark_loss,
            'vd_SNR': avg_valid_snr,
            'vd_BER': avg_valid_ber
        })

    del valid_loss
    del valid_audio_loss
    del valid_rst_audio_loss
    del valid_contrast_loss
    del valid_watermark_loss
    del valid_snr
    del valid_ber
    gc.collect()

    # Print average validation results
    print()
    print(f'Validation avg:    '
          f'Val. Loss'
          f' Au. Loss'
          f' rAu.Loss'
          f' rAu.TriL'
          f' Wm. Loss'
          f'      SNR'
          f'      BER')
    print(f'                    '
          f'{avg_valid_loss:8.4f} '
          f'{avg_valid_audio_loss:8.4f} '
          f'{avg_valid_rst_audio_loss:8.4f} '
          f'{avg_valid_contrast_loss:8.4f} '
          f'{avg_valid_watermark_loss:8.4f} '
          f'{avg_valid_snr:8.4f} '
          f'{avg_valid_ber:8.4f} ')
    print()

    # Reset model to training mode
    model.train()

    return avg_valid_loss, avg_valid_audio_loss, avg_valid_watermark_loss, avg_valid_snr, avg_valid_ber, avg_valid_rst_audio_loss, avg_valid_contrast_loss
