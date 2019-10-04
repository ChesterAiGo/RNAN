try: from Assignment2.Ours.network import *
except: from network import *
try: from Assignment2.Ours.dataset import *
except: from dataset import *
try: from Assignment2.Ours.utils import *
except: from utils import *



if __name__ == '__main__':

    # change to the path where the model weight is stored
    saved_model_path = 'saved_models/q37_5_RNAN_random_33.01889196449235.path.tar'

    # mode determines either the q40 data or q37 data is loaded for evaluation
    mode = 'q37'

    # build and load model onto GPU
    model = Network(8, 64, 3)
    model.load_state_dict(torch.load(saved_model_path))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # build data loader
    q37_p, q40_p, src_p = 'ELEC5306_DATA/input_vimeo_part_hevc_qp37_crop/', 'ELEC5306_DATA/vimeo_part_q40_crop/', 'ELEC5306_DATA/vimeo_part_crop/'
    train_txt_p, val_txt_p = 'ELEC5306_DATA/temp_sep_trainlist.txt', 'ELEC5306_DATA/temp_sep_validationlist.txt'
    D = Dataset(q37_p, q40_p, src_p, train_txt_p, val_txt_p)

    # start evaluation
    with torch.no_grad():
        print("-- Evaluation has begun")
        model.eval()

        # load testing data
        X, Y = D.load_test_data(mode=mode)
        X, Y = np.asarray(X, dtype=np.float32) / 255.0, np.asarray(Y, dtype=np.float32) / 255.0
        X, Y = X.transpose((0, 3, 1, 2)), Y.transpose((0, 3, 1, 2))
        tensor_X = torch.stack([torch.Tensor(_) for _ in X])
        tensor_Y = torch.stack([torch.Tensor(_) for _ in Y])

        tmp_dataset = data.TensorDataset(tensor_X, tensor_Y)
        tmp_dataloader = data.DataLoader(tmp_dataset, batch_size=1)

        # saving mses/psnrs for all images
        # note that X_src contains the metric values between raw data (i.e. q40/q37) and labels
        # while X_out contains the metric values between the restored data and labels
        mse_l_src, mse_l_out, psnr_l_src, psnr_l_out = [], [], [], []

        # evaluate
        for n_count, batch_yx in enumerate(tmp_dataloader):
            # Load batch data and send to GPU
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

            mse_l_src.append(mse_src)
            mse_l_out.append(mse_out)
            psnr_l_src.append(psnr_src)
            psnr_l_out.append(psnr_out)

            print("-- Evaluating for image:", n_count + 1, ', MSE:', mse_out, ', PSNR:', psnr_out)


        print("-- MSE_src:", np.mean(mse_l_src), flush=True)
        print("-- MSE_reconstructed:", np.mean(mse_l_out), ",max:", np.max(mse_l_out), ",min:", np.min(mse_l_out), flush=True)
        print("-- PSNR_src:", np.mean(psnr_l_src), flush=True)
        print("-- PSNR_reconstructed:", np.mean(psnr_l_out),  ",max:", np.max(psnr_l_out), ",min:", np.min(psnr_l_out), flush=True)
