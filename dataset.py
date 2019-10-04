import os
import numpy as np
import PIL.Image as Image


# class Dataset():
#     def __init__(self):
#         # define paths, changable
#         self.train_Y_path, self.test_Y_path = 'data/train_label/', 'data/test_label/'
#         self.train_q40_path, self.test_q40_path = 'data/train_data_q40/', 'data/test_data_q40/'
#         self.train_q37_path, self.test_q37_path = 'data/train_data_q37/', 'data/test_data_q37/'
#
#
#     # Load a batch of X and Y
#     def load_train_data(self, mode='q40', batch_size=None, preprocessing='src', cropped_w=None, cropped_h=None):
#         if   (mode == 'q37'): tmp_X_path = self.train_q37_path
#         elif (mode == 'q40'): tmp_X_path = self.train_q40_path
#         else: raise Exception("Unknown mode")
#         tmp_Y_path = self.train_Y_path
#
#         assert self.count_valid_imgs(tmp_X_path) == 4802 and self.count_valid_imgs(tmp_Y_path) == 4802
#
#         # to further store the actual loaded images
#         X, Y = [], []
#         # acquire paths of images
#         x_list, y_list = self.get_img_list(tmp_X_path), self.get_img_list(tmp_Y_path)
#
#         # also supports to randomly extract a batch, for testing purpose
#         if (batch_size is not None and batch_size <= 4802):
#             rnd_idx = np.random.choice(np.arange(4802), replace=False, size=(batch_size, ))
#             x_list = np.array(x_list)[rnd_idx]
#             y_list = np.array(y_list)[rnd_idx]
#
#         # load training data (q37 or q40)
#         rnd_seed = 0
#         for each_x in x_list:
#             # using full image for training
#             if(preprocessing == 'src'):
#                 X.append(np.asarray(Image.open(each_x).convert("RGB"), dtype=np.uint8))
#
#             # using center part only for training
#             elif(preprocessing == 'center'):
#                 loaded_img = Image.open(each_x)
#                 width, height = loaded_img.size
#                 cropped = loaded_img.crop((width / 4, height / 4, 3 * width / 4, 3 * height / 4))
#                 cropped = np.asarray(cropped.convert("RGB"), dtype=np.uint8)
#                 X.append(cropped)
#
#             # randomly crop a part for training
#             elif(preprocessing == 'random' and cropped_w is not None and cropped_h is not None):
#                 loaded_img = Image.open(each_x)
#                 width, height = loaded_img.size
#                 assert cropped_w > int(width / 2) and cropped_h > int(height / 2)
#                 np.random.seed(rnd_seed)
#                 rnd_w, rnd_h = int(np.random.randint(0, width - cropped_w)), int(
#                     np.random.randint(0, height - cropped_h))
#                 cropped = loaded_img.crop((rnd_w, rnd_h, rnd_w + cropped_w, rnd_h + cropped_h))
#                 cropped = np.asarray(cropped.convert("RGB"), dtype=np.uint8)
#                 X.append(cropped)
#
#             rnd_seed += 1
#
#         # load original data (labels)
#         rnd_seed = 0
#         for each_y in y_list:
#             # using full image for training
#             if (preprocessing == 'src'):
#                 Y.append(np.asarray(Image.open(each_y), dtype=np.uint8))
#
#             # using center part only for training
#             elif (preprocessing == 'center'):
#                 loaded_img = Image.open(each_y)
#                 width, height = loaded_img.size
#                 cropped = loaded_img.crop((width / 4, height / 4, 3 * width / 4, 3 * height / 4))
#                 cropped = np.asarray(cropped.convert("RGB"), dtype=np.uint8)
#                 Y.append(cropped)
#
#             # randomly crop a part for training
#             elif (preprocessing == 'random' and cropped_w is not None and cropped_h is not None):
#                 loaded_img = Image.open(each_y)
#                 width, height = loaded_img.size
#                 assert cropped_w > int(width / 2) and cropped_h > int(height / 2)
#                 np.random.seed(rnd_seed)
#                 rnd_w, rnd_h = int(np.random.randint(0, width - cropped_w)), int(
#                     np.random.randint(0, height - cropped_h))
#                 cropped = loaded_img.crop((rnd_w, rnd_h, rnd_w + cropped_w, rnd_h + cropped_h))
#                 cropped = np.asarray(cropped.convert("RGB"), dtype=np.uint8)
#                 Y.append(cropped)
#
#             rnd_seed += 1
#
#
#         assert len(X) == len(Y)
#
#         # change to uint8 dtype to save space
#         X, Y = np.array(X, dtype=np.uint8), np.array(Y, dtype=np.uint8)
#         return X, Y
#
#
#     # Load all testing data
#     def load_test_data(self, mode='q40'):
#         if   (mode == 'q37'): tmp_X_path = self.test_q37_path
#         elif (mode == 'q40'): tmp_X_path = self.test_q40_path
#         else: raise Exception("Unknown mode")
#         tmp_Y_path = self.test_Y_path
#
#         assert self.count_valid_imgs(tmp_X_path) == 616 and self.count_valid_imgs(tmp_Y_path) == 616
#
#         # to further store the actual loaded images
#         X, Y = [], []
#         # acquire paths of images
#         x_list, y_list = self.get_img_list(tmp_X_path), self.get_img_list(tmp_Y_path)
#
#         # load q40/q37 data
#         for each_x in x_list: X.append(np.asarray(Image.open(each_x), dtype=np.uint8))
#         # load labels
#         for each_y in y_list: Y.append(np.asarray(Image.open(each_y), dtype=np.uint8))
#
#         assert len(X) == len(Y)
#
#         # change to uint8 dtype to save space
#         X, Y = np.array(X, dtype=np.uint8), np.array(Y, dtype=np.uint8)
#         return X, Y
#
#
#     # this function counts the number of valid images to ensure the entire dataset exists
#     # there are 4802 training samples and 616 validation samples
#     def count_valid_imgs(self, tgt_path, ext_names=['jpg', 'png']):
#         cnt = 0
#         for each_img in os.listdir(tgt_path):
#             if(each_img.split(".")[-1] in ext_names): cnt += 1
#
#         return cnt
#
#     # this function returns the list of all image paths under tgt_path, with ext name png or jpg
#     # it is noteworthy that we renamed the images (added an index) to better realize 1-to-1 mapping
#     def get_img_list(self, tgt_path):
#         tmp_img_list = [(x, int(x.split('_')[0])) for x in os.listdir(tgt_path) if 'png' in x or 'jpg' in x]
#         sorted_list = sorted(tmp_img_list, key=lambda x: x[1])
#         return [tgt_path + x[0] for x in sorted_list]


class Dataset():
    def __init__(self, q37_p, q40_p, src_p, train_txt_p, val_txt_p):
        # define paths, changable
        self.q37_path, self.q40_path, self.src_path = q37_p, q40_p, src_p
        self.train_txt_path, self.val_txt_path = train_txt_p, val_txt_p

    # Load a batch of X and Y
    def load_train_data(self, mode='q40', batch_size=None, preprocessing='src', cropped_w=None, cropped_h=None):
        if   (mode == 'q37'): tmp_X_path = self.q37_path
        elif (mode == 'q40'): tmp_X_path = self.q40_path
        else: raise Exception("Unknown mode")
        tmp_Y_path = self.src_path

        # to further store the actual loaded images
        X, Y = [], []
        # acquire paths of images
        x_list, y_list = self.get_img_list(tmp_X_path, self.train_txt_path, mode), self.get_img_list(tmp_Y_path, self.train_txt_path, 'src')

        # also supports to randomly extract a batch, for testing purpose
        if (batch_size is not None and batch_size <= 4802):
            rnd_idx = np.random.choice(np.arange(4802), replace=False, size=(batch_size, ))
            x_list = np.array(x_list)[rnd_idx]
            y_list = np.array(y_list)[rnd_idx]

        # load training data (q37 or q40)
        rnd_seed = 0
        for each_x in x_list:
            # using full image for training
            if(preprocessing == 'src'):
                X.append(np.asarray(Image.open(each_x).convert("RGB"), dtype=np.uint8))

            # using center part only for training
            elif(preprocessing == 'center'):
                loaded_img = Image.open(each_x)
                width, height = loaded_img.size
                cropped = loaded_img.crop((width / 4, height / 4, 3 * width / 4, 3 * height / 4))
                cropped = np.asarray(cropped.convert("RGB"), dtype=np.uint8)
                X.append(cropped)

            # randomly crop a part for training
            elif(preprocessing == 'random' and cropped_w is not None and cropped_h is not None):
                loaded_img = Image.open(each_x)
                width, height = loaded_img.size
                assert cropped_w >= int(width / 2) and cropped_h >= int(height / 2)
                np.random.seed(rnd_seed)
                rnd_w, rnd_h = int(np.random.randint(0, width - cropped_w)), int(
                    np.random.randint(0, height - cropped_h))
                cropped = loaded_img.crop((rnd_w, rnd_h, rnd_w + cropped_w, rnd_h + cropped_h))
                cropped = np.asarray(cropped.convert("RGB"), dtype=np.uint8)
                X.append(cropped)

            rnd_seed += 1

        # load original data (labels)
        rnd_seed = 0
        for each_y in y_list:
            # using full image for training
            if (preprocessing == 'src'):
                Y.append(np.asarray(Image.open(each_y), dtype=np.uint8))

            # using center part only for training
            elif (preprocessing == 'center'):
                loaded_img = Image.open(each_y)
                width, height = loaded_img.size
                cropped = loaded_img.crop((width / 4, height / 4, 3 * width / 4, 3 * height / 4))
                cropped = np.asarray(cropped.convert("RGB"), dtype=np.uint8)
                Y.append(cropped)

            # randomly crop a part for training
            elif (preprocessing == 'random' and cropped_w is not None and cropped_h is not None):
                loaded_img = Image.open(each_y)
                width, height = loaded_img.size
                assert cropped_w >= int(width / 2) and cropped_h >= int(height / 2)
                np.random.seed(rnd_seed)
                rnd_w, rnd_h = int(np.random.randint(0, width - cropped_w)), int(
                    np.random.randint(0, height - cropped_h))
                cropped = loaded_img.crop((rnd_w, rnd_h, rnd_w + cropped_w, rnd_h + cropped_h))
                cropped = np.asarray(cropped.convert("RGB"), dtype=np.uint8)
                Y.append(cropped)

            rnd_seed += 1


        assert len(X) == len(Y)

        # change to uint8 dtype to save space
        X, Y = np.array(X, dtype=np.uint8), np.array(Y, dtype=np.uint8)
        return X, Y


    # Load all testing data
    def load_test_data(self, mode='q40'):
        if   (mode == 'q37'): tmp_X_path = self.q37_path
        elif (mode == 'q40'): tmp_X_path = self.q40_path
        else: raise Exception("Unknown mode")
        tmp_Y_path = self.src_path

        # to further store the actual loaded images
        X, Y = [], []
        # acquire paths of images
        x_list, y_list = self.get_img_list(tmp_X_path, self.val_txt_path, mode), self.get_img_list(tmp_Y_path, self.val_txt_path, 'src')

        # load q40/q37 data
        for each_x in x_list: X.append(np.asarray(Image.open(each_x), dtype=np.uint8))
        # load labels
        for each_y in y_list: Y.append(np.asarray(Image.open(each_y), dtype=np.uint8))

        assert len(X) == len(Y)

        # change to uint8 dtype to save space
        X, Y = np.array(X, dtype=np.uint8), np.array(Y, dtype=np.uint8)
        return X, Y



    # this function returns the list of all image paths under tgt_path, with ext name png or jpg
    # it is noteworthy that we renamed the images (added an index) to better realize 1-to-1 mapping
    def get_img_list(self, tgt_path, txt_path, source):

        result = []
        with open(txt_path, 'r') as f:
            all_folders = [x.strip() for x in f.readlines()]
            for each_folder in all_folders:
                folder_path = tgt_path + each_folder + "/"

                for idx in range(1, 8):
                    if(source == 'src'):
                        result.append(folder_path + 'im' + str(idx) + ".png")

                    if(source == 'q40'):
                        result.append(folder_path + 'im_q40_' + str(idx) + ".jpg")

                    if(source == 'q37'):
                        result.append(folder_path + 'im_q37_' + str(idx) + ".png")

        return result





