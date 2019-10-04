

try: from Assignment2.Ours.network import *
except: from network import *
try: from Assignment2.Ours.dataset import *
except: from dataset import *
try: from Assignment2.Ours.utils import *
except: from utils import *


if __name__ == '__main__':

    epochs = 50
    mode = 'q40' # q37 or q40

    # Build models
    model = Network(depth=8, n_maps=64, n_channel=3)
    # discriminator = Discriminator()
    model_name = 'RNAN'

    # Place model on GPU
    # Note that this code works fine on a single K80 (12gb memory) GPU card
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.train()
    # discriminator.to(device)
    # discriminator.train()

    # Define optimizers
    optimizer = optim.Adam(model.parameters())
    # optimizer_D = optim.Adam(discriminator.parameters())

    # Build dataset loader
    print("-- Loading data")
    q37_p, q40_p, src_p = 'ELEC5306_DATA/input_vimeo_part_hevc_qp37_crop/', 'ELEC5306_DATA/vimeo_part_q40_crop/', 'ELEC5306_DATA/vimeo_part_crop/'
    train_txt_p, val_txt_p = 'ELEC5306_DATA/temp_sep_trainlist.txt', 'ELEC5306_DATA/temp_sep_validationlist.txt'
    D = Dataset(q37_p, q40_p, src_p, train_txt_p, val_txt_p)

    # Define parameters
    batch_size = 5
    preprocessing = 'random'
    cropped_w, cropped_h = 112, 64
    alpha, beta, gamma = 1.0, 0.001, 0
    model_save_to = 'saved_models/' + mode + '_' + str(batch_size) + '_' + model_name + '_' + preprocessing + '_.pth.tar'

    # Define losses
    ssim_object = SSIM_loss()
    # binary_cross_entropy_object = nn.BCELoss()

    # Load training data
    X, Y = D.load_train_data(mode=mode, preprocessing=preprocessing, cropped_h=cropped_h, cropped_w=cropped_w, batch_size=4802)
    X, Y = np.asarray(X, dtype=np.float32) / 255.0, np.asarray(Y, dtype=np.float32) / 255.0
    X, Y = X.transpose((0, 3, 1, 2)), Y.transpose((0, 3, 1, 2))

    tensor_X = torch.stack([torch.Tensor(_) for _ in X])
    tensor_Y = torch.stack([torch.Tensor(_) for _ in Y])

    tmp_dataset = data.TensorDataset(tensor_X, tensor_Y)
    tmp_dataloader_train = data.DataLoader(tmp_dataset, batch_size=batch_size)

    # Load testing data
    X, Y = D.load_test_data(mode=mode)
    X, Y = np.asarray(X, dtype=np.float32) / 255.0, np.asarray(Y, dtype=np.float32) / 255.0
    X, Y = X.transpose((0, 3, 1, 2)), Y.transpose((0, 3, 1, 2))
    tensor_X = torch.stack([torch.Tensor(_) for _ in X])
    tensor_Y = torch.stack([torch.Tensor(_) for _ in Y])

    tmp_dataset = data.TensorDataset(tensor_X, tensor_Y)
    tmp_dataloader_test = data.DataLoader(tmp_dataset, batch_size=1)

    # Remove used tensors to maximize available pytorch memory
    del X, Y, tensor_X, tensor_Y, tmp_dataset
    torch.cuda.empty_cache()

    # Start training
    best_psnr = 0
    for e in range(epochs):

        start_T = time()
        print("-- Epochs:", e, flush=True)

        # set model to train mode
        model.train()

        for n_count, batch_yx in enumerate(tmp_dataloader_train):
            # Load batch data and send to GPU
            batch_x, batch_y = batch_yx[1], batch_yx[0]
            batch_x, batch_y = batch_x.cuda(), batch_y.cuda()
            batch_x.to(device)
            batch_y.to(device)
            reconstructed = model(batch_x)

            # Train discriminator first
            # optimizer_D.zero_grad()
            # label_ones = torch.full((batch_x.size()[0],), 1, device=device)

            # real_bce_loss = binary_cross_entropy_object(discriminator(batch_y), label_ones)
            # zero-GAN
            # real_bce_loss = binary_cross_entropy_object(discriminator(torch.full(batch_y.size(), 0, device=device)), label_ones)
            # real_bce_loss.backward()

            # label_zeros = torch.full((batch_x.size()[0],), 0, device=device)
            # reconstructed_bce_loss = binary_cross_entropy_object(discriminator(reconstructed.detach()), label_zeros)
            # zero-GAN
            # reconstructed_bce_loss = binary_cross_entropy_object(discriminator(reconstructed.detach() - batch_y), label_zeros)
            # reconstructed_bce_loss.backward()
            # optimizer_D.step()

            # Then train Our improved RNAN network
            optimizer.zero_grad()
            # illusion_bce_loss = binary_cross_entropy_object(discriminator(reconstructed), label_ones)
            # zero-GAN
            # illusion_bce_loss = binary_cross_entropy_object(discriminator(reconstructed - batch_y), label_ones)
            mse_loss = torch.nn.functional.mse_loss(reconstructed, batch_y)
            ssim_loss = -ssim_object(reconstructed, batch_y)
            loss = alpha * mse_loss + beta * ssim_loss # + gamma * illusion_bce_loss
            # weighted samples
            loss *= (mse_loss / mse_loss.sum()) # -> opt 1
            # loss *= ((mse_loss - torch.min(mse_loss)) / (torch.max(mse_loss) - torch.min(mse_loss))) # -> opt2

            print("-- Loss of epoch", e, 'in iteration', n_count, ',MSE =', mse_loss.item(), ',SSIM =',
                  ssim_loss.item(), flush=True) #',GAN loss =', illusion_bce_loss.item(), flush=True)

            loss.backward()
            optimizer.step()

            # remove for more available gpu memory
            del batch_x, batch_y, reconstructed, mse_loss, ssim_loss, loss # illusion_bce_loss, , label_ones, label_zeros, real_bce_loss, reconstructed_bce_loss
            torch.cuda.empty_cache()


        # no_grad can reduce the memory requriement significantly
        with torch.no_grad():
            # set model to evaluation model
            model.eval()

            # saving mses/psnrs for all images
            # note that X_src contains the metric values between raw data (i.e. q40/q37) and labels
            # while X_out contains the metric values between the restored data and labels
            mse_l_src, mse_l_out, psnr_l_src, psnr_l_out = [], [], [], []

            # start evaluation, 1 image each time due to memory limit
            for n_count, batch_yx in enumerate(tmp_dataloader_test):
                batch_x, batch_y = batch_yx[1].cuda(), batch_yx[0].cuda()

                # calculate mse/psnr between the restored data and labels
                batch_y.to(device)
                reconstructed = model(batch_x)
                mse_out = math.sqrt(torch.mean((reconstructed - batch_y) ** 2.0))
                psnr_out = calculate_psnr(mse_out)
                del reconstructed
                torch.cuda.empty_cache()

                # calculate mse/psnr between the raw data and labels
                batch_x.to(device)
                mse_src = math.sqrt(torch.mean((batch_x - batch_y) ** 2.0))
                psnr_src = calculate_psnr(mse_src)
                del batch_x, batch_y
                torch.cuda.empty_cache()

                # append the values into lists
                mse_l_src.append(mse_src)
                mse_l_out.append(mse_out)
                psnr_l_src.append(psnr_src)
                psnr_l_out.append(psnr_out)

                print("-- Evaluating for image:", n_count + 1, ', MSE:', mse_out, ', PSNR:', psnr_out)

            # Print result for each epoch
            print(e, "-- MSE_src:", np.mean(mse_l_src), ",max:", np.max(mse_l_src), ",min:", np.min(mse_l_src), flush=True)
            print(e, "-- MSE_reconstructed:", np.mean(mse_l_out), ",max:", np.max(mse_l_out), ",min:", np.min(mse_l_out), flush=True)
            print(e, "-- PSNR_src:", np.mean(psnr_l_src), ",max:", np.max(psnr_l_src), ",min:", np.min(psnr_l_src), flush=True)
            print(e, "-- PSNR_reconstructed:", np.mean(psnr_l_out), ",max:", np.max(psnr_l_out), ",min:",np.min(psnr_l_out), flush=True)
            print(e, "-- Time taken:", round(time() - start_T, 4), 'seconds\n', flush=True)

            # save the best model
            if (np.mean(psnr_l_out) > best_psnr):
                best_psnr = np.mean(psnr_l_out)
                print("-- Saving to file, ", best_psnr, model_save_to[:-8] + str(best_psnr) + ".path.tar")
                torch.save(model.state_dict(), model_save_to[:-8] + str(best_psnr) + ".path.tar")


