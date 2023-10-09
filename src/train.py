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
from torch_stft import STFT
from pydtw import SoftDTW
from pystct import sdct_torch, isdct_torch
from losses import ssim, SNR, PSNR, StegoLoss, calc_ber, signal_noise_ratio
from visualization import viz2paper, viz4seq


def save_checkpoint(state, is_best, filename=os.path.join(os.environ.get('OUT_PATH'), 'checkpoint.pt')):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print("=> Saving a new best model")
        print(f'SAVING TO: {filename}')
        torch.save(state, filename)  # save checkpoint
    else:
        print("=> Loss did not improve")


def train(model, tr_loader, vd_loader, beta, lam, lr, epochs=5, val_itvl=500, val_size=50, prev_epoch=None, prev_i=None,
          summary=None, slide=50, experiment=0, transform='cosine', stft_small=True, ft_container='mag', thet=0,
          dtw=False):
    # Initialize wandb logs
    wandb.init(project='PixInWavRGB')
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

    ini = time.time()
    best_loss = np.inf

    # Initialize waveform loss constructor
    if dtw:
        softDTW = SoftDTW(gamma=1.0, normalize=True)
    else:
        l1wavLoss = nn.L1Loss()

    # Initialize STFT transform constructor
    if transform == 'fourier':
        stft = STFT(
            filter_length=2 ** 11 - 1 if stft_small else 2 ** 12 - 1,
            hop_length=132 if stft_small else 66,
            win_length=2 ** 11 - 1 if stft_small else 2 ** 12 - 1,
            window='hann'
        ).to(device)
        stft.num_samples = 67522

    # Start training ...
    for epoch in range(epochs):

        if prev_epoch != None and epoch < prev_epoch - 1: continue  # Checkpoint pass

        # Initialize training metrics storage
        train_loss, train_loss_cover, train_loss_secret, train_loss_spectrum, snr, psnr, ssim_secret, train_wav_loss = [], [], [], [], [], [], [], []
        vd_loss, vd_loss_cover, vd_loss_secret, vd_snr, vd_psnr, vd_ssim, vd_wav = [], [], [], [], [], [], []
        train_ber, vd_ber = [], []

        # Print headers for the losses
        print()
        print(' Iter.     Time  Tr. Loss  Au. MSE   Im. WF  Spectr.  Au. SNR Im. PSNR Im. SSIM   Au. WF BER')
        for i, data in enumerate(tr_loader):

            if prev_i != None and i < prev_i - 1: continue  # Checkpoint pass

            # Load data from the loader
            # (B,secret_len), (B,secret_len,), (B,L)  B=1
            secrets, secrets_bin, covers = data[0][0].to(device), data[0][1].to(device), data[1].to(device)
            secrets = secrets.type(torch.cuda.FloatTensor)

            optimizer.zero_grad()

            # Forward through the model
            # (B,N,T,C), (B,N,T,C), (B,L), (B,secret_len)
            cover_fft, containers_fft, container_wav, revealed = model(secrets, covers)

            # Compute the loss
            cover_fft = cover_fft.squeeze(0)
            containers_fft = containers_fft.squeeze(0)
            original_wav = covers.squeeze(0)
            container_wav = container_wav.squeeze(0)
            # container_2x = stft.transform(container_wav)[0].unsqueeze(1)  # TODO: obsismc: why have this?
            # TODO: what is loss_spectrum
            # print(
                # f"before loss: cover_fft shape: {cover_fft.shape}, containters_fft shape: {containers_fft.shape}, "
                # f"secrets shape:{secrets.shape}, revealed shape:{revealed.shape}, original_wav: {original_wav.shape},"
                # f"container_wav: {container_wav.shape}")
            loss, loss_cover, loss_secret, loss_spectrum = StegoLoss(secrets, cover_fft, containers_fft, None,
                                                                     revealed, beta)

            # Compute waveform loss. Add it only if specified
            # if dtw:
            #     wav_loss = softDTW(original_wav.cpu().unsqueeze(0), container_wav.cpu().unsqueeze(0))
            #     if transform == 'fourier': wav_loss = wav_loss[0]
            # else:
            #     wav_loss = l1wavLoss(original_wav.cpu().unsqueeze(0), container_wav.cpu().unsqueeze(0))
            # objective_loss = loss + lam * wav_loss
            wav_loss = l1wavLoss(original_wav.cpu().unsqueeze(0), container_wav.cpu().unsqueeze(0))
            objective_loss = loss + lam * wav_loss

            with torch.autograd.set_detect_anomaly(True):
                objective_loss.backward()
            optimizer.step()

            # Compute audio and image metrics
            # if (transform != 'fourier') or (ft_container != 'magphase'):
            #     containers_phase = None  # Otherwise it's the phase container
            # snr_audio = SNR(
            #     covers,
            #     containers,
            #     None if transform == 'cosine' else phase,
            #     containers_phase,
            #     transform=transform,
            #     transform_constructor=None if transform == 'cosine' else stft,
            #     ft_container=ft_container,
            # )
            snr_audio = signal_noise_ratio(original_wav.cpu().detach().numpy(), container_wav.cpu().detach().numpy())
            ber = calc_ber(revealed, secrets)
            psnr_image = torch.tensor(0.0)  # PSNR(secrets, revealed)
            ssim_image = torch.tensor(0.0)  # ssim(secrets, revealed)

            # Append and average the new losses
            train_loss.append(loss.detach().item())
            train_loss_cover.append(loss_cover.detach().item())
            train_loss_secret.append(loss_secret.detach().item())
            train_loss_spectrum.append(loss_spectrum.detach().item())
            snr.append(snr_audio)
            train_ber.append(ber.item())
            psnr.append(psnr_image.detach().item())
            ssim_secret.append(ssim_image.detach().item())
            train_wav_loss.append(wav_loss.detach().item())

            avg_train_loss = np.mean(train_loss[-slide:])
            avg_train_loss_cover = np.mean(train_loss_cover[-slide:])
            avg_train_loss_secret = np.mean(train_loss_secret[-slide:])
            avg_train_loss_spectrum = np.mean(train_loss_spectrum[-slide:])
            avg_snr = np.mean(snr[-slide:])
            avg_ber = np.mean(train_ber[-slide:])
            avg_ssim = np.mean(ssim_secret[-slide:])
            avg_psnr = np.mean(psnr[-slide:])
            avg_wav_loss = np.mean(train_wav_loss[-slide:])

            print('(#%4d)[%5d s] %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f' % (
                i, np.round(time.time() - ini), loss.detach().item(), loss_cover.detach().item(),
                loss_secret.detach().item(), loss_spectrum.detach().item(), snr_audio, psnr_image.detach().item(),
                ssim_image.detach().item(), wav_loss.detach().item(), ber.item()))

            # Log train average loss to wandb
            wandb.log({
                'tr_i_loss': avg_train_loss,
                'tr_i_cover_loss': avg_train_loss_cover,
                'tr_i_secret_loss': avg_train_loss_secret,
                'tr_i_spectrum_loss': avg_train_loss_spectrum,
                'tr_SNR': avg_snr,
                'tr_BER': avg_ber,
                'tr_PSNR': avg_psnr,
                'tr_SSIM': avg_ssim,
                'tr_wav': avg_wav_loss,
            })

            # Every 'val_itvl' iterations, do a validation step
            if (i % val_itvl == 0) and (i != 0):
                criterion = softDTW if dtw else l1wavLoss
                avg_valid_loss, avg_valid_loss_cover, avg_valid_loss_secret, avg_valid_snr, avg_valid_psnr, avg_valid_ssim, avg_valid_wav, avg_valid_ber = validate(
                    model, vd_loader, beta, val_size=val_size, transform=transform,
                    transform_constructor=stft if transform == 'fourier' else None, ft_container=ft_container,
                    wav_criterion=criterion, tr_i=i, epoch=epoch)

                vd_loss.append(avg_valid_loss)
                vd_loss_cover.append(avg_valid_loss_cover)
                vd_loss_secret.append(avg_valid_loss_secret)
                vd_snr.append(avg_valid_snr)
                vd_ber.append(avg_valid_ber)
                vd_psnr.append(avg_valid_psnr)
                vd_ssim.append(avg_valid_ssim)
                vd_wav.append(avg_valid_wav)

                is_best = bool(avg_valid_loss < best_loss)
                # Save checkpoint if is a new best
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'beta': beta,
                    'lr': lr,
                    'i': i + 1,
                    'tr_loss': train_loss,
                    'tr_cover_loss': train_loss_cover,
                    'tr_loss_secret': train_loss_secret,
                    'tr_snr': snr,
                    'tr_ber': train_ber,
                    'tr_psnr': psnr,
                    'tr_ssim': ssim_secret,
                    'tr_wav': train_wav_loss,
                    'vd_loss': vd_loss,
                    'vd_cover_loss': vd_loss_cover,
                    'vd_loss_secret': vd_loss_secret,
                    'vd_snr': vd_snr,
                    'vd_ber': vd_ber,
                    'vd_psnr': vd_psnr,
                    'vd_ssim': vd_ssim,
                    'vd_wav': vd_wav,
                }, is_best=is_best, filename=os.path.join(os.environ.get('OUT_PATH'),
                                                          f'{experiment}-{summary}/{epoch + 1}-{experiment}-{summary}.pt'))

                # Print headers again to resume training
                print()
                print(' Iter.     Time  Tr. Loss  Au. MSE   Im. WF  Spectr.  Au. SNR Im. PSNR Im. SSIM   Au. WF BER')

        # Print average validation results after every epoch
        print()
        print('Epoch average:      Tr. Loss  Au. MSE   Im. WF  Spectr.  Au. SNR Im. PSNR Im. SSIM   Au. WF  BER')
        print('Epoch %1d/%1d:          %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f' % (
            epoch + 1, epochs, avg_train_loss, avg_train_loss_cover, avg_train_loss_secret, avg_train_loss_spectrum,
            avg_snr, avg_psnr, avg_ssim, avg_wav_loss, avg_ber))
        print()

        # Log train average loss to wandb
        wandb.log({
            'avg_tr_loss': avg_train_loss,
            'avg_tr_cover_loss': avg_train_loss_cover,
            'avg_tr_secret_loss': avg_train_loss_secret,
            'avg_tr_spectrum_loss': avg_train_loss_spectrum,
            'avg_tr_SNR': avg_snr,
            'avg_tr_ber': avg_ber,
            'avg_tr_PSNR': avg_psnr,
            'avg_tr_SSIM': avg_ssim,
            'avg_tr_WF': avg_wav_loss
        })

        is_best = bool(avg_train_loss < best_loss)
        best_loss = min(avg_train_loss, best_loss)

        # Save checkpoint if is a new best
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_loss': best_loss,
            'beta': beta,
            'lr': lr,
            'i': i + 1,
        }, is_best=is_best, filename=os.path.join(os.environ.get('OUT_PATH'),
                                                  f'{experiment}-{summary}/{epoch + 1}-{experiment}-{summary}.pt'))

    print(f"Training took {time.time() - ini} seconds")
    torch.save(model.state_dict(),
               os.path.join(os.environ.get('OUT_PATH'), f'{experiment}-{summary}/final_run_{experiment}.pt'))
    return model, avg_train_loss


def validate(model, vd_loader, beta, val_size=50, transform='cosine', transform_constructor=None, ft_container='mag',
             wav_criterion=None, epoch=None, tr_i=None, thet=0):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parallelize on GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model).cuda()

    model.to(device)

    # Set to evaluation mode
    model.eval()
    loss = 0

    valid_loss, valid_loss_cover, valid_loss_secret, valid_loss_spectrum, valid_snr, valid_psnr, valid_ssim, valid_wav = [], [], [], [], [], [], [], []
    valid_ber = []
    vd_datalen = len(vd_loader)

    # Start validating ...
    iniv = time.time()
    with torch.no_grad():
        for i, data in enumerate(vd_loader):

            # Load data from the loader
            # (B,secret_len), (B,secret_len,), (B,L)  B=1
            secrets, secrets_bin, covers = data[0][0].to(device), data[0][1].to(device), data[1].to(device)
            secrets = secrets.type(torch.cuda.FloatTensor)

            # Forward through the model
            # (B,N,T,2), (B,N,T,2), (B,L), (B,secret_len)
            cover_fft, containers_fft, container_wav, revealed = model(secrets, covers)

            # Visualize results
            if i == 0:
                fig = viz4seq(secrets[0].cpu(), revealed[0].cpu(), cover_fft.cpu(), containers_fft.cpu(), None, None,
                              transform, ft_container)
                wandb.log({f"Revelation at epoch {epoch}, vd iteration {tr_i}": fig})

            # Compute the loss
            cover_fft = cover_fft.squeeze(0)
            containers_fft = containers_fft.squeeze(0)
            original_wav = covers.squeeze(0)
            container_wav = container_wav.squeeze(0)
            # container_2x = transform_constructor.transform(container_wav)[0].unsqueeze(1)
            loss, loss_cover, loss_secret, loss_spectrum = StegoLoss(secrets, cover_fft, containers_fft, None,
                                                                     revealed, beta)

            # Compute audio and image metrics
            # if (transform != 'fourier') or (ft_container != 'magphase'):
            #     containers_phase = None  # Otherwise it's the phase container
            vd_snr_audio = signal_noise_ratio(original_wav.cpu().detach().numpy(), container_wav.cpu().detach().numpy())
            ber = calc_ber(revealed, secrets)
            vd_psnr_image = torch.tensor(0.0)  # PSNR(secrets, revealed)
            ssim_image = torch.tensor(0.0)  # ssim(secrets, revealed)

            if wav_criterion is not None:
                # if transform == 'cosine':
                #     original_wav = isdct_torch(covers.squeeze(0).squeeze(0), frame_length=4096, frame_step=130,
                #                                window=torch.hamming_window)
                # elif transform == 'fourier':
                #     original_wav = transform_constructor.inverse(covers.squeeze(1), phase.squeeze(1))
                wav_loss = wav_criterion(original_wav.cpu().unsqueeze(0), container_wav.cpu().unsqueeze(0))

            valid_loss.append(loss.detach().item())
            valid_loss_cover.append(loss_cover.detach().item())
            valid_loss_secret.append(loss_secret.detach().item())
            valid_loss_spectrum.append(loss_spectrum.detach().item())
            valid_snr.append(vd_snr_audio)
            valid_ber.append(ber.item())
            valid_psnr.append(vd_psnr_image.detach().item())
            valid_ssim.append(ssim_image.detach().item())
            valid_wav.append(wav_loss.detach().item())

            # Stop validation after val_size steps
            if i >= val_size: break

        avg_valid_loss = np.mean(valid_loss)
        avg_valid_loss_cover = np.mean(valid_loss_cover)
        avg_valid_loss_secret = np.mean(valid_loss_secret)
        avg_valid_loss_spectrum = np.mean(valid_loss_spectrum)
        avg_valid_snr = np.mean(valid_snr)
        avg_valid_ber = np.mean(valid_ber)
        avg_valid_psnr = np.mean(valid_psnr)
        avg_valid_ssim = np.mean(valid_ssim)
        avg_valid_wav = np.mean(valid_wav)

        wandb.log({
            'vd_loss': avg_valid_loss,
            'vd_cover_loss': avg_valid_loss_cover,
            'vd_secret_loss': avg_valid_loss_secret,
            'vd_spectrum_loss': avg_valid_loss_spectrum,
            'vd_SNR': avg_valid_snr,
            'vd_BER': avg_valid_ber,
            'vd_PSNR': avg_valid_psnr,
            'vd_SSIM': avg_valid_ssim,
            'vd_WF': avg_valid_wav
        })

    del valid_loss
    del valid_loss_cover
    del valid_loss_secret
    del valid_loss_spectrum
    del valid_snr
    del valid_ber
    del valid_psnr
    del valid_ssim
    del valid_wav
    gc.collect()

    # Print average validation results
    print()
    print('Validation avg:    Val. Loss    Cover   Secret  Au. SNR Im. PSNR Im. SSIM   Au. WF   BER')
    print('                    %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f %8.4f' % (
        avg_valid_loss, avg_valid_loss_cover, avg_valid_loss_secret, avg_valid_snr, avg_valid_psnr, avg_valid_ssim,
        avg_valid_wav, avg_valid_ber))
    print()

    # Reset model to training mode
    model.train()

    return avg_valid_loss, avg_valid_loss_cover, avg_valid_loss_secret, avg_valid_snr, avg_valid_psnr, avg_valid_ssim, avg_valid_wav, avg_valid_ber
