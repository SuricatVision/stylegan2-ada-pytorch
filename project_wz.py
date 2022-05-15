import os
import sys

import legacy

sys.path.insert(0, '/home/misha/AdmireMirror/data-tools/face_alignment')
sys.path.insert(1, '/home/misha/AdmireMirror/stylegan2_pytorch')

import glob
import math

from functools import partial
import dnnlib
import cv2
import lpips
import numpy as np
import torch
import torch.nn as nn
import torch_optimizer as optim
from PIL import Image
from criteria.id_loss import IDLoss
from torch.nn import functional as F
from torchvision import transforms
from tqdm import tqdm
from tdffa.TDDFA import TDDFA
from futils.utils import align_face, get_eyes_landmarks, BGSegmentator, get_tdffa_landmarks
from face_boxes import FaceBoxes
# from utils.functions import draw_landmarks
import yaml


def resize_img(img_gen):
    batch, channel, height, width = img_gen.shape

    if height > 256:
        factor = height // 256

        img_gen = img_gen.reshape(
            batch, channel, height // factor, factor, width // factor, factor
        )
        img_gen = img_gen.mean([3, 5])
    return img_gen


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def get_square(pt, scale=1.6):
    scale /= 2
    pt_m = (pt[0] + pt[1]) / 2
    s = np.abs(pt[1] - pt[0]).max()
    bb = [pt_m[0] - scale * s, pt_m[1] - scale * s, pt_m[0] + scale * s, pt_m[1] + scale * s]
    return np.array(bb, int)


def create_mask(im, pt):
    mask = np.zeros_like(im, dtype=np.uint8)
    points = [[17, 36], [26, 45], [36, 39], [39, 42], [42, 45], [48, 54], [27, 33], [57, 8], [4, 6], [5, 7], [9, 11], [10, 12]]
    scales = [1.8, 1.8, 1.8, 1.1, 1.8, 1.15, 1.25, 1.15, 0.75, 0.75, 0.75, 0.75]
    for k in range(len(scales)):
        rect = get_square(pt[points[k]], scale=scales[k])
        mask[rect[1]:rect[3], rect[0]:rect[2]] = 1

    return mask


def make_image(tensor):
    return (
        tensor.detach()
            .clamp_(min=-1, max=1)
            .add(1)
            .div_(2)
            .mul(255)
            .type(torch.uint8)
            .permute(0, 2, 3, 1)
            .to("cpu")
            .squeeze(dim=0)
            .numpy()
    )


def mean_latent(model, n_latent, style_dim=512, device="cuda:0"):
    latent_in = torch.randn(
        n_latent, style_dim, device=device
    )
    latent = model.mapping(latent_in, None).mean(0, keepdim=True)

    return latent


class ProjectStylegan(nn.Module):
    # ToDo load all styles in one directory
    def __init__(self, pkl_path, device='cuda:0', size=512, face_id=False,
                 source_mean_path='', w_start=8, w_end=8, step=70,
                 ckpt_id='', tddfa='',
                 styles='',
                 ):
        super(ProjectStylegan, self).__init__()

        self.face_id = face_id
        self.device = device
        self.size = size
        self.w_start = w_start
        self.w_end = w_end
        self.lr_rampdown = 0.25
        self.lr_rampup = 0.05
        self.lr = 1.5  # for NovoGard
        self.step = step
        resize = min(self.size, 256)

        self.transform = transforms.Compose(
            [
                transforms.Resize(resize),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        if self.face_id:
            self.id_loss = IDLoss(ckpt_id).to(device).eval()

        if isinstance(tddfa, str):
            with torch.no_grad():
                cfg = yaml.load(open(tddfa), Loader=yaml.SafeLoader)
                self.tddfa = TDDFA(gpu_mode=True, **cfg)
        else:
            self.tddfa = tddfa

        with dnnlib.util.open_url(pkl_path) as f:
            self.g_ema = legacy.load_network_pkl(f)["G_ema"]
        self.g_ema.eval()
        self.g_ema = self.g_ema.to(device)

        self.source_mean = torch.load(source_mean_path)
        self.percept = lpips.PerceptualLoss(
            model="net-lin", net="vgg", use_gpu=device.startswith("cuda"),  # squeeze
        )
        self.latent_mean_all = mean_latent(self.g_ema, 40000).detach()
        # latent_z_in = latent_mean.clone()

        ### load styles ####
        names = glob.glob(styles + '/*.npy')
        self.names = names
        print([os.path.basename(x) for x in names])
        self.v_styles = []
        for i in range(len(names)):  # ix:
            # for i in ix:
            path = names[i]
            # shutil.copy(path,path.replace('mix2_choose','v3'))

            v1 = np.load(path)
            v1 = torch.tensor(v1, dtype=torch.float32, device=device)
            self.v_styles.append(v1.clone())
        print(len(self.v_styles))

    def set_style(self, latent_w):
        imgs = []
        a = 4
        for i, vn in enumerate(self.v_styles):  # ToDo batch all shifts
            latent_new = latent_w.clone()
            latent_new[:, self.w_start:] = (latent_w[:, self.w_start:] + a * vn[:, self.w_start:]) / (1 + a)
            img_gen = self.g_ema.synthesis(latent_new, noise_mode="const")
            img_ar = make_image(img_gen)
            imgs.append(img_ar.copy())

        return imgs

    def set_centroid_style(self, latent_w, start_n=4):
        imgs = []
        vn = self.v_styles[0]
        A = [0, 1, 1.5, 2, 2.5, 5, 10, 10, 10]
        starts = [start_n, start_n, start_n, start_n, start_n, start_n, start_n, start_n - 1, start_n - 2]
        for i, a in enumerate(A):
            latent_new = latent_w.clone()
            latent_new[:, starts[i]:] = (latent_w[:, starts[i]:] + a * vn[:, starts[i]:]) / (1 + a)
            img_gen = self.g_ema.synthesis(latent_new, noise_mode="const")
            img_ar = make_image(img_gen)
            imgs.append(img_ar.copy())

        return imgs

    def set_styles_diff_layers(self, latent_w):
        data = []
        for i_vn, vn in enumerate(self.v_styles):
            imgs = []
            starts = [i for i in range(latent_w.shape[1])]
            os.makedirs(os.path.join("generated_images/style_analysis_network-snapshot-004600/diff_layers_same_coeff", str(i_vn)))
            # A = [0, 0.5, 1, 1.5, 2, 0.2, 10, 3.5, 4, 4.5, 5, 5.5, 6, 6.5]
            A = [4 for _ in range(14)]
            for i, a in enumerate(A):
                print(i)
                latent_new = latent_w.clone()
                latent_new[:, starts[i]:] = (latent_w[:, starts[i]:] + a * vn[:, starts[i]:]) / (1 + a)
                img_gen = self.g_ema.synthesis(latent_new, noise_mode="const")
                img_ar = make_image(img_gen)
                img_ar = cv2.cvtColor(img_ar, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join("generated_images/style_analysis_network-snapshot-004600/diff_layers_same_coeff", str(i_vn), str(i) + ".jpg"), img_ar)
                imgs.append(img_ar.copy())
            data.append(imgs)

        return data

    def set_style_same_layer_diff_coeff(self, latent_w, n_layer=4):
        data = []
        for vn in self.v_styles:
            imgs = []
            A = [0, 0.5, 1.0, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
            for i, a in enumerate(A):
                latent_new = latent_w.clone()
                latent_new[:, n_layer:] = (latent_w[:, n_layer:] + a * vn[:, n_layer:]) / (1 + a)
                img_gen = self.g_ema.synthesis(latent_new, noise_mode="const")
                img_ar = make_image(img_gen)
                imgs.append(img_ar.copy())
            data.append(imgs)

        return data

    def new_set_style(self, latent_w):
        imgs = []
        self.w_start = 0
        a = torch.tensor([0, 0, 0.0, 0.0, 0.0, 4, 4, 4, 4, 4, 4, 4, 4, 4], dtype=torch.float32).cuda().unsqueeze(0).unsqueeze(2)
        for i, vn in enumerate(self.v_styles):  # ToDo batch all shifts
            latent_new = (latent_w + a * vn) / (1 + a)
            img_gen = self.g_ema.synthesis(latent_new, noise_mode="const")
            img_ar = make_image(img_gen)
            imgs.append(img_ar.copy())

        return imgs

    def trackbar_set_style(self, latent_w, n_layer, value):
        value /= 100
        self.w_start = 0
        a = torch.tensor([0.05, 0.0, 0.025, 0.05, 0.0, 0.0, 0.1, 0.1, 1, 0.5, 0, 1, 1, 5], dtype=torch.float32).cuda()
        a[n_layer-1] = value
        a = a.unsqueeze(0).unsqueeze(2)
        vn = self.v_styles[1]
        latent_new = (latent_w + a * vn) / (1 + a)
        img_gen = self.g_ema.synthesis(latent_new, noise_mode="const")
        img_ar = make_image(img_gen)
        img_ar = cv2.cvtColor(img_ar, cv2.COLOR_RGB2BGR)
        cv2.imshow("Image", img_ar)

    def save_styles_layerwise(self, latent_w, num_ws, base_dir="generated_images/style_analysis_network-snapshot-004600"):
        values = [i / 10 for i in range(51)]
        for i, vn in enumerate(self.v_styles):
            os.makedirs(os.path.join(base_dir, str(i)), exist_ok=True)
            for w_i in range(num_ws):
                os.makedirs(os.path.join(base_dir, str(i), str(w_i)), exist_ok=True)
                for value in values:
                    path = os.path.join(base_dir, str(i), str(w_i), str(value) + ".jpg")
                    c = torch.zeros(num_ws, dtype=torch.float32).cuda()
                    c[w_i] = value
                    c = c.unsqueeze(0).unsqueeze(2)
                    latent_new = (latent_w + c * vn) / (1 + c)
                    img_gen = self.g_ema.synthesis(latent_new, noise_mode="const")
                    img_ar = make_image(img_gen)
                    img_ar = cv2.cvtColor(img_ar, cv2.COLOR_RGB2BGR)
                    cv2.imwrite(path, img_ar)

    def projection(self, img):
        img0 = img.copy()
        img = self.transform(Image.fromarray(img[:, :, ::-1]))
        imgs = img[None, :].to(self.device)

        if self.face_id:
            feat_real = self.id_loss.extract_feats(imgs).detach()

        boxes = 4 * np.array([[32, 35, 220, 223]])

        with torch.no_grad():
            param_lst, roi_box_lst = self.tddfa(img0, boxes)
        ver_lst = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
        ver_lst = ver_lst[0].astype(int)
        mask = create_mask(img0, ver_lst.transpose(1, 0))
        mask = cv2.resize(mask, (256, 256))
        mask = mask.transpose(2, 0, 1)
        mask = torch.from_numpy(mask).type(torch.float32).to(self.device)  # .type(torch.bool)
        # mask=1+4*mask
        mask = mask[None, :]  # for batch size=1

        n_start = self.w_start
        n_end = self.w_end
        # latent_w_in = self.source_mean['latent_w0'][None, :n_end].to(self.device)
        # latent_z_in = self.source_mean['latent_z'][None, :].to(self.device)
        latent_w_in = torch.randn(1, 14, 64)[:, :n_end, :].to(self.device)
        latent_z_in = torch.randn(1, 512).to(self.device)
        latent_w_in.requires_grad = True
        latent_z_in.requires_grad = True

        optimizer = optim.NovoGrad([latent_z_in, latent_w_in], lr=self.lr)
        truncation = 0.5
        pbar = tqdm(range(int(1.75 * self.step)))
        step = self.step
        latent_path = []
        for i in pbar:

            if i > self.step - 2 and loss_id > 0.08:
                step = int(1.5 * self.step)  # 1.5
            if i > 1.5 * self.step - 2 and loss_id > 0.08:
                step = int(1.75 * self.step)  # 1.5
            if i > self.step - 2 and loss_id < 0.08:
                break
            t = i / step

            lr = get_lr(t, self.lr)
            optimizer.param_groups[0]["lr"] = lr
            latent_w0 = self.g_ema.mapping(latent_z_in, None)[:, n_start:, :]
            # latent_w0 = latent_w0.unsqueeze(1).repeat(1, self.g_ema.n_latent - n_start, 1)

            latent_w_part1 = self.latent_mean_all[:, :n_end] + truncation * (latent_w_in[:, :n_end] - self.latent_mean_all[:, :n_end])
            latent_w_part2 = self.latent_mean_all[:, n_start:] + truncation * (latent_w0 - self.latent_mean_all[:, n_start:])
            latent_w_plus = torch.cat([latent_w_part1[:, :n_start], latent_w_part2], dim=1)

            if n_start != n_end:
                # latent_w_plus[:,n_start:n_end]=latent_w_part2[:,:n_end-n_start] + truncation * (latent_w_part1[:,n_start:n_end] - latent_w_part2[:,:n_end-n_start])
                latent_w_plus[:, n_start:n_end] = latent_w_part2[:, :n_end - n_start] + latent_w_part1[:, n_start:n_end]

            img_gen = self.g_ema.synthesis(latent_w_plus, noise_mode="const")
            img_gen = resize_img(img_gen)
            loss = 0
            p_loss = self.percept(img_gen, imgs).sum()
            loss += p_loss
            if self.face_id and (i % 2 == 0 or i > 0.7 * self.step):
                loss_id = 0
                feat_gen = self.id_loss.extract_feats(img_gen)
                for k in range(img_gen.shape[0]):
                    loss_id += 1 - feat_gen[k].dot(feat_real[k])
                loss += 0.3 * loss_id  # 0.2* #0.1
            if np.random.rand() > 0.65:
                img_gen_mask = mask * img_gen.clone()
                imgs_mask = mask * imgs.clone()
                p_loss_mask = self.percept(img_gen_mask, imgs_mask).sum()
                loss += 3 * p_loss_mask

            loss_reg = 0.0002 * torch.norm(latent_w0)
            loss_reg += 0.001 * torch.norm(latent_w_in)  # 0.01
            if n_start != n_end:
                loss_reg += 0.003 * torch.norm(latent_w_in[:, 8:n_end])  # 0.01
            loss += loss_reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(
                (
                    f"perceptual: {p_loss.item():.4f};"
                    f" reg: {loss_reg.item():.4f};"
                    # f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
                    f"loss_id: {loss_id.item():.4f};"
                    # f"lr {lr:.4f}"
                )
            )

        img_gen = self.g_ema.synthesis(latent_w_plus.detach(), noise_mode="const")

        img_ar = make_image(img_gen)
        return latent_w_plus, img_ar

    def projection_w_plus(self, img):
        img0 = img.copy()
        img = self.transform(Image.fromarray(img[:, :, ::-1]))
        imgs = img[None, :].to(self.device)

        if self.face_id:
            feat_real = self.id_loss.extract_feats(imgs).detach()

        boxes = 4 * np.array([[32, 35, 220, 223]])

        with torch.no_grad():
            param_lst, roi_box_lst = self.tddfa(img0, boxes)
        ver_lst = self.tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)
        ver_lst = ver_lst[0].astype(int)
        mask = create_mask(img0, ver_lst.transpose(1, 0))
        mask = cv2.resize(mask, (256, 256))
        mask = mask.transpose(2, 0, 1)
        mask = torch.from_numpy(mask).type(torch.float32).to(self.device)  # .type(torch.bool)
        # mask=1+4*mask
        mask = mask[None, :]  # for batch size=1

        # latent_w_in = self.source_mean['latent_w0'][None, :n_end].to(self.device)
        # latent_z_in = self.source_mean['latent_z'][None, :].to(self.device)
        latent_w_in = torch.randn(1, 14, 256).to(self.device)
        latent_w_in.requires_grad = True

        optimizer = optim.NovoGrad([latent_w_in], lr=self.lr)
        truncation = 0.5
        pbar = tqdm(range(int(1.75 * self.step)))
        step = self.step
        latent_path = []
        for i in pbar:

            if i > self.step - 2 and loss_id > 0.08:
                step = int(1.5 * self.step)  # 1.5
            if i > 1.5 * self.step - 2 and loss_id > 0.08:
                step = int(1.75 * self.step)  # 1.5
            if i > self.step - 2 and loss_id < 0.08:
                break
            t = i / step

            lr = get_lr(t, self.lr)
            optimizer.param_groups[0]["lr"] = lr
            # latent_w0 = latent_w0.unsqueeze(1).repeat(1, self.g_ema.n_latent - n_start, 1)

            latent_w_plus = self.latent_mean_all + truncation * (latent_w_in - self.latent_mean_all)

            img_gen = self.g_ema.synthesis(latent_w_plus, noise_mode="const")
            img_gen = resize_img(img_gen)
            loss = 0
            p_loss = self.percept(img_gen, imgs).sum()
            loss += p_loss
            if self.face_id and (i % 2 == 0 or i > 0.7 * self.step):
                loss_id = 0
                feat_gen = self.id_loss.extract_feats(img_gen)
                for k in range(img_gen.shape[0]):
                    loss_id += 1 - feat_gen[k].dot(feat_real[k])
                loss += 0.3 * loss_id  # 0.2* #0.1
            if np.random.rand() > 0.65:
                img_gen_mask = mask * img_gen.clone()
                imgs_mask = mask * imgs.clone()
                p_loss_mask = self.percept(img_gen_mask, imgs_mask).sum()
                loss += 3 * p_loss_mask

            # loss_reg = 0.0002 * torch.norm(latent_w0)
            loss_reg = 0.001 * torch.norm(latent_w_in)  # 0.01
            # if n_start != n_end:
            #     loss_reg += 0.003 * torch.norm(latent_w_in[:, 8:n_end])  # 0.01
            loss += loss_reg
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.set_description(
                (
                    f"perceptual: {p_loss.item():.4f};"
                    f" reg: {loss_reg.item():.4f};"
                    # f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
                    f"loss_id: {loss_id.item():.4f};"
                    # f"lr {lr:.4f}"
                )
            )

        img_gen = self.g_ema.synthesis(latent_w_plus.detach(), noise_mode="const")

        img_ar = make_image(img_gen)
        return latent_w_plus, img_ar

    def forward(self, img):
        latent_w_plus, img_proj = self.projection(img)
        if len(self.v_styles) > 1:
            imgs = self.set_style(latent_w_plus)
        else:
            imgs = self.set_centroid_style(latent_w_plus)
        imgs.append(img_proj[0])
        return imgs


class FaceAligner(torch.nn.Module):

    def __init__(self, path, tddfa_path):
        super().__init__()
        # with torch.no_grad():
        #     cfg = yaml.load(open(tddfa_path), Loader=yaml.SafeLoader)
        #     self.tddfa = TDDFA(gpu_mode=False, **cfg)
        self.face_detector = FaceBoxes(path)
        self.face_size = 512

    def face_preproc(self, img, tddfa):
        with torch.no_grad():
            landmarks = get_tdffa_landmarks(img, self.face_detector, tddfa, min_face=-1, skip_angle=-1,
                                            skip_face_th=0.7)
            aligned = align_face(img,
                                 landmarks,
                                 output_size=self.face_size,
                                 transform_size=int(self.face_size * 1.6),
                                 enable_padding=False
                                 )

        return aligned


def create_trackbar(latent_w, projector):
    func = partial(projector.trackbar_set_style, latent_w, 3)
    cv2.namedWindow("Image")
    cv2.createTrackbar("TrackBar", "Image", 0, 500, func)

    func(0)

    cv2.waitKey()


if __name__ == "__main__":
    from PIL import Image

    projector = ProjectStylegan("/home/misha/AdmireMirror/stylegan2-ada-pytorch/train_logs/00001-watercolor_mix-stylegan2-resumeffhq1024/network-snapshot-004600.pkl",
                                  styles="/home/misha/AdmireMirror/stylegan2-ada-pytorch/generated_images/styles_network-snapshot-004600",
                                  source_mean_path="/home/misha/AdmireMirror/stylegan2_pytorch/data/ffhq_mix2_512_wz_dual_8_codes_mean.pt",
                                  tddfa="/home/misha/AdmireMirror/data-tools/face_alignment/tdffa/configs/mb1_120x120.yml",
                                  face_id=True,
                                  ckpt_id="/home/misha/AdmireMirror/stylegan2_pytorch/data/model_ir_se50.pth",
                                  step=280,
                                w_start=12,
                                w_end=12)
    aligner = FaceAligner("/home/misha/AdmireMirror/stylegan2_pytorch/data/FaceBoxesProd.pth",
                          "/home/misha/AdmireMirror/data-tools/face_alignment/tdffa/configs/mb1_120x120.yml")

    # image = cv2.imread("/home/misha/Desktop/MyPhotos/1.jpg")
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image_pil = Image.fromarray(image)
    # aligned_image = aligner.face_preproc(image_pil, projector.tddfa)
    # aligned_image = cv2.cvtColor(np.array(aligned_image.copy()), cv2.COLOR_RGB2BGR)
    # aligned_image = cv2.resize(aligned_image, (256, 256))
    # cv2.imwrite("aligned_image.jpg", aligned_image)
    # w_plus, projected_image = projector.projection_w_plus(aligned_image)
    # projected_image = cv2.cvtColor(projected_image, cv2.COLOR_RGB2BGR)
    # cv2.imwrite("projected_image.jpg", projected_image)
    # with open("projected_latent.npy", "wb") as f:
    #     np.save(f, w_plus.cpu().detach().numpy())

    projected_image = cv2.imread("projected_image.jpg")
    cv2.imshow("Image", projected_image)
    cv2.waitKey(0)
    projected_image = cv2.cvtColor(projected_image, cv2.COLOR_BGR2RGB)
    w_plus = torch.from_numpy(np.load("projected_latent.npy")).cuda()

    # create_trackbar(w_plus, projector)
    # projector.save_styles_layerwise(w_plus, 14)
    # stylized_image = projector.set_style(w_plus)[0]
    stylized_images = projector.new_set_style(w_plus)
    for i, stylized_image in enumerate(stylized_images):
        print(i+1, projector.names[i])
        stylized_image = cv2.cvtColor(stylized_image, cv2.COLOR_RGB2BGR)
        cv2.imshow("Image", stylized_image)
        cv2.waitKey(0)
    # for ii, d in enumerate(stylized_images):
    #     for i, stylized_image in enumerate(d):
    #         print(i+1, projector.names[ii])
    #         stylized_image = cv2.cvtColor(stylized_image, cv2.COLOR_RGB2BGR)
    #         cv2.imshow("Image", stylized_image)
    #         cv2.waitKey(0)
