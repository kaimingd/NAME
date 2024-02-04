#!/usr/bin/env python
# coding: utf-8


import torch
import torchvision
import os
import glob
import time 
import pickle
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from PIL import ImageDraw
from src.data2 import LungDataset, blend, Pad, Crop, Resize
from src.models import UNet, PretrainedUNet
from src.metrics import jaccard, dice

from skimage.metrics import peak_signal_noise_ratio
from networks.vit_seg_modeling import VisionTransformer as ViT_seg
from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg

add = "minus1"
dir_root = "paired_data/adapter_chestxray_paired"
# dir_root = "/hhd/1/dkm/LCDG/work_dir_diffuser/sketch/adapter_chestxray_unseen1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_folder = Path("input", "dataset")
origins_folder = Path(dir_root)
masks_folder = Path(dir_root+"_"+add)
#/hhd/1/dkm/LCDG/work_dir_diffuser/test2/adapter_chestxray_example15_imgimg_minus1
#/hhd/1/dkm/LCDG/work_dir_diffuser/test2/adapter_chestxray_example15_imgimg_minus1_binary
#/hhd/1/dkm/LCDG/work_dir_diffuser/test2/adapter_chestxray_example15_imgmask_mul1
#/hhd/1/dkm/LCDG/work_dir_diffuser/test2/adapter_chestxray_example15_imgmask_mul1_binary
#/hhd/1/dkm/LCDG/work_dir_diffuser/test2/adapter_chestxray_example15_imgmask_mul2
#/hhd/1/dkm/LCDG/work_dir_diffuser/test2/adapter_chestxray_example15_imgmask_mul2_binary
# models_folder = Path("models")
images_folder = Path("images_transunet_"+add)
if not os.path.exists(images_folder):
    os.makedirs(images_folder)
batch_size = 1

def load():
# ## Data loading
    origins_list = [f.stem for f in origins_folder.glob("*.jpg")]
    origin_mask_list = [f.stem for f in masks_folder.glob("*.jpg")]
    print(len(origins_list))
    print(len(origin_mask_list))


    # origin_mask_list = [(mask_name.replace("_mask", ""), mask_name) for mask_name in masks_list]
    split_file = "splits_transunet_simu_chexpert.pk"
    if os.path.isfile(split_file):
        with open("splits_transunet_simu_chexpert.pk", "rb") as f:
            splits = pickle.load(f)
    else:
        splits = {}
        splits["train"], splits["test"] = train_test_split(origin_mask_list, test_size=0.2, random_state=42) #0.2
        splits["train"], splits["val"] = train_test_split(splits["train"], test_size=0.1, random_state=42) #0.1

        with open("splits_transunet_simu_chexpert.pk", "wb") as f:
            pickle.dump(splits, f)


    val_test_transforms = torchvision.transforms.Compose([
        Resize((512, 512)),
    ])

    train_transforms = torchvision.transforms.Compose([
        Pad(200),
        # Crop(300),
        val_test_transforms,
    ])

    datasets = {x: LungDataset(
        splits[x], 
        origins_folder, 
        masks_folder, 
        train_transforms if x == "train" else val_test_transforms
    ) for x in ["train", "test", "val"]}

    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size) for x in ["train", "test", "val"]}

    idx = 0
    phase = "train"

    plt.figure(figsize=(20, 10))
    origin, mask, _ = datasets[phase][idx]
    pil_origin = torchvision.transforms.functional.to_pil_image(origin + 0.5).convert("RGB")
    # pil_mask = torchvision.transforms.functional.to_pil_image(mask.float())
    pil_mask = torchvision.transforms.functional.to_pil_image(mask + 0.5).convert("RGB")

    plt.subplot(1, 3, 1)
    plt.title("origin image")
    plt.imshow(np.array(pil_origin))

    plt.subplot(1, 3, 2)
    plt.title("manually labeled mask")
    plt.imshow(np.array(pil_mask))

    # plt.subplot(1, 3, 3)
    # plt.title("blended origin + mask")
    # plt.imshow(np.array(blend(origin, mask)));

    plt.savefig(images_folder / "data-example.png", bbox_inches='tight')
    return datasets,dataloaders


# ## Model training
# unet = UNet(in_channels=1, out_channels=2, batch_norm=True)
def train(datasets,dataloaders):
    # unet = PretrainedUNet(
    #     in_channels=1,
    #     out_channels=2, 
    #     batch_norm=True, 
    #     upscale_mode="bilinear"
    # )
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = 1
    config_vit.n_skip = 3
    # if args.vit_name.find('R50') != -1:
    config_vit.patches.grid = (int(512 / 16), int(512 / 16))
    unet = ViT_seg(config_vit, img_size=512, num_classes=config_vit.n_classes).cuda()


    unet = unet.to(device)
    # optimizer = torch.optim.SGD(unet.parameters(), lr=0.0005, momentum=0.9)
    optimizer = torch.optim.Adam(unet.parameters(), lr=0.0005)
    criterion = torch.nn.MSELoss()
    train_log_filename = "train-log.txt"
    epochs = 100
    best_val_loss = np.inf
    model_name = "unet-6v.pt"     #需要修改

    hist = []

    for e in range(epochs):
        start_t = time.time()
        
        print("train phase")
        unet.train()
        train_loss = 0.0
        for origins, masks, name in dataloaders["train"]:
            num = origins.size(0)
            
            origins = origins.to(device)#[4,1,512,512]
            masks = masks.to(device)#[4,512,512]
            
            optimizer.zero_grad()
            
            outs = unet(origins) #[1,1,512,512]
            # softmax = torch.nn.functional.log_softmax(outs, dim=1)#[4,2,512,512]
            # loss = torch.nn.MSELoss(outs, masks)
            # loss = torch.nn.functional.nll_loss(softmax,masks)
            loss = criterion(outs,masks)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * num
            # break
            
        train_loss = train_loss / len(datasets['train'])
        
        print("validation phase")
        unet.eval()
        val_loss = 0.0
        val_jaccard = 0.0
        val_dice = 0.0

        for origins, masks,name in dataloaders["val"]:
            num = origins.size(0)

            origins = origins.to(device)
            masks = masks.to(device)

            with torch.no_grad():
                outs = unet(origins)
                softmax = torch.nn.functional.log_softmax(outs, dim=1)
                # val_loss +=  criterion((outs[:,0]+outs[:,1])/2,masks[:,0]).item() * num
                val_loss +=  criterion(outs,masks).item() * num

                # val_loss += torch.nn.functional.nll_loss(softmax, masks).item() * num

                outs = torch.argmax(softmax, dim=1)
                outs = outs.float()
                masks = masks.float()
                # val_jaccard += jaccard(masks, outs.float()).item() * num
                # val_dice += dice(masks, outs).item() * num

            # print(".", end="")
            # break
            
        val_loss = val_loss / len(datasets["val"])
        # val_jaccard = val_jaccard / len(datasets["val"])
        # val_dice = val_dice / len(datasets["val"])
        val_jaccard = 0
        val_dice = 0
        
        end_t = time.time()
        spended_t = end_t - start_t
        
        with open(images_folder / train_log_filename, "a") as train_log_file:
            report = f"epoch: {e+1}/{epochs}, time: {spended_t}, train loss: {train_loss}, \n"\
                + f"val loss: {val_loss}, val jaccard: {val_jaccard}, val dice: {val_dice}"

            hist.append({
                "time": spended_t,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_jaccard": val_jaccard,
                "val_dice": val_dice,
            })

            print(report)
            train_log_file.write(report + "\n")
            
            torch.save(unet.state_dict(), images_folder / model_name.replace('-6v','-6v-latest'))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(unet.state_dict(), images_folder / model_name.replace('-6v','-6v-best'))
                print("model saved")
                train_log_file.write("model saved\n")
            
            
    plt.figure(figsize=(15,7))
    train_loss_hist = [h["train_loss"] for h in hist]
    plt.plot(range(len(hist)), train_loss_hist, "b", label="train loss")

    val_loss_hist = [h["val_loss"] for h in hist]
    plt.plot(range(len(hist)), val_loss_hist, "r", label="val loss")

    val_dice_hist = [h["val_dice"] for h in hist]
    plt.plot(range(len(hist)), val_dice_hist, "g", label="val dice")

    val_jaccard_hist = [h["val_jaccard"] for h in hist]
    plt.plot(range(len(hist)), val_jaccard_hist, "y", label="val jaccard")

    plt.legend()
    plt.xlabel("epoch")
    plt.savefig(images_folder / model_name.replace(".pt", "-train-hist.png"))

    time_hist = [h["time"] for h in hist]
    overall_time = sum(time_hist) // 60
    mean_epoch_time = sum(time_hist) / len(hist)
    print(f"epochs: {len(hist)}, overall time: {overall_time}m, mean epoch time: {mean_epoch_time}s")


    torch.cuda.empty_cache()


# ## Evaluate

def val(datasets,dataloaders):
    # unet = PretrainedUNet(1, 2, True, "bilinear")

    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = 1
    config_vit.n_skip = 3
    # if args.vit_name.find('R50') != -1:
    config_vit.patches.grid = (int(512 / 16), int(512 / 16))
    unet = ViT_seg(config_vit, img_size=512, num_classes=config_vit.n_classes).cuda()
    criterion = torch.nn.MSELoss()
    model_name = "unet-6v.pt".replace('-6v','-6v-latest')
    unet.load_state_dict(torch.load(images_folder / model_name, map_location=torch.device("cpu")))
    unet.to(device)
    unet.eval();
    
    
    dataset_id = "generated_test_from_chexpert"
    images_folder_gt = Path("images_transunet_"+add,dataset_id,"gt")
    images_folder_img = Path("images_transunet_"+add,dataset_id,"img")
    images_folder_pred = Path("images_transunet_"+add,dataset_id,"pred")
    images_folder_mask = Path("images_transunet_"+add,dataset_id,"mask")
    if not os.path.exists(images_folder_gt):
        os.makedirs(images_folder_gt)
    if not os.path.exists(images_folder_img):
        os.makedirs(images_folder_img)
    if not os.path.exists(images_folder_pred):
        os.makedirs(images_folder_pred)
    if not os.path.exists(images_folder_mask):
        os.makedirs(images_folder_mask)
        
    test_loss = 0.0
    test_jaccard = 0.0
    test_dice = 0.0
    test_psnr = 0.0

    for origins, masks, name in dataloaders["test"]:
        num = origins.size(0)

        origins = origins.to(device)
        masks = masks.to(device)

        with torch.no_grad():
            outs = unet(origins)
            softmax = torch.nn.functional.log_softmax(outs, dim=1)
            # test_loss +=  criterion((outs[:,0]+outs[:,1])/2,masks[:,0]).item() * num

            test_loss +=  criterion(outs,masks).item() * num
            # test_loss += torch.nn.functional.nll_loss(softmax, masks).item() * num

            # outs = torch.argmax(softmax, dim=1)
            out = outs.float()
            # out = (outs[:,0]+outs[:,1])/2
            masks = masks.float()
            # test_jaccard += jaccard(masks, outs).item() * num
            # test_dice += dice(masks, outs).item() * num

            pil_origin = torchvision.transforms.functional.to_pil_image(origins[0] + 0.5).convert("RGB")
            pil_mask_gt = torchvision.transforms.functional.to_pil_image(masks[0] + 0.5).convert("RGB")
            pil_mask_pred = torchvision.transforms.functional.to_pil_image(out[0] + 0.5).convert("RGB")
                    
            pil_mask_pred = pil_mask_pred.convert("L")
            pil_mask_gt = pil_mask_gt.convert("L")
            
            image_array = np.array(pil_mask_pred)

            white_mask = np.where(image_array >= 240)
            image_array[white_mask] = 0
            pil_mask_pred = Image.fromarray(image_array)            
            pil_mask = pil_mask_pred.point(lambda p: p > 35 and 255)     
                        
            pil_origin.save(images_folder_img / name[0])
            pil_mask_gt.save(images_folder_gt / name[0])
            pil_mask_pred.save(images_folder_pred / name[0])
            pil_mask.save(images_folder_mask / name[0])
            # plt.subplot(1, 3, 1)
            # plt.title("origin image")
            # plt.imshow(np.array(pil_origin))

            # plt.subplot(1, 3, 2)
            # plt.title("manually mask_pred")
            # plt.imshow(np.array(pil_mask_pred))
            
            # plt.subplot(1, 3, 3)
            # plt.title("manually mask_gt")
            # plt.imshow(np.array(pil_mask_gt))
            
            # plt.savefig(images_folder_mix / name[0], bbox_inches='tight')
            # test_psnr += peak_signal_noise_ratio(np.array(pil_mask_pred), np.array(pil_mask_gt)) * num



        print(".", end="")

    # test_loss = test_loss / len(datasets["test"])
    # # test_jaccard = test_jaccard / len(datasets["test"])
    # test_psnr = test_psnr / len(datasets["test"])
    
    # test_jaccard = 0
    # test_psnr = 0

    # print()
    # print(f"avg test loss: {test_loss}")
    # print(f"avg test jaccard: {test_jaccard}")
    # print(f"avg test dice: {test_psnr}")


    # num_samples = 9
    # phase = "test"

    # subset = torch.utils.data.Subset(
    #     datasets[phase], 
    #     np.random.randint(0, len(datasets[phase]), num_samples)
    # )
    # random_samples_loader = torch.utils.data.DataLoader(subset, batch_size=1)
    # plt.figure(figsize=(20, 25))

    # for idx, (origin, mask) in enumerate(random_samples_loader):
    #     plt.subplot((num_samples // 3) + 1, 3, idx+1)

    #     origin = origin.to(device)
    #     mask = mask.to(device)

    #     with torch.no_grad():
    #         out = unet(origin)
    #         softmax = torch.nn.functional.log_softmax(out, dim=1)
    #         out = torch.argmax(softmax, dim=1)

    #         jaccard_score = jaccard(mask.float(), out.float()).item()
    #         dice_score = dice(mask.float(), out.float()).item()

    #         origin = origin[0].to("cpu")
    #         out = out[0].to("cpu")
    #         mask = mask[0].to("cpu")

    #         plt.imshow(np.array(blend(origin, mask, out)))
    #         plt.title(f"jaccard: {jaccard_score:.4f}, dice: {dice_score:.4f}")
    #         print(".", end="")
                
    # plt.savefig(images_folder / "obtained-results.png", bbox_inches='tight')
    # print()         
    # print("red area - predict")
    # print("green area - ground truth")
    # print("yellow area - intersection")
        
def val_real_chexpert():
    # unet = PretrainedUNet(1, 2, True, "bilinear")
    
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = 1
    config_vit.n_skip = 3
    # if args.vit_name.find('R50') != -1:
    config_vit.patches.grid = (int(512 / 16), int(512 / 16))
    unet = ViT_seg(config_vit, img_size=512, num_classes=config_vit.n_classes).cuda()
    model_name = "unet-6v-latest.pt"
    unet.load_state_dict(torch.load(images_folder / model_name, map_location=torch.device("cpu")))
    unet.to(device)
    unet.eval();


    img_root = "images_transunet_minus1/chexpert_test/img"

    dataset_id = "chexpert_test"
    images_folder_gt = Path("images_transunet_"+add,dataset_id,"gt")
    images_folder_img = Path("images_transunet_"+add,dataset_id,"img")
    images_folder_pred = Path("images_transunet_"+add,dataset_id,"pred")
    images_folder_mask = Path("images_transunet_"+add,dataset_id,"mask")
    # images_folder_mix = Path("images_transunet_"+add,dataset_id,"mix")
    if not os.path.exists(images_folder_gt):
        os.makedirs(images_folder_gt)
    if not os.path.exists(images_folder_img):
        os.makedirs(images_folder_img)
    if not os.path.exists(images_folder_pred):
        os.makedirs(images_folder_pred)
    if not os.path.exists(images_folder_mask):
        os.makedirs(images_folder_mask)
    # if not os.path.exists(images_folder_mix):
    #     os.makedirs(images_folder_mix)
        
    img_list = os.listdir(images_folder_img)
    img_list.sort()
    for i,img_item in enumerate(img_list):
        img_dir = os.path.join(img_root,img_item)
        gt = Image.open(img_dir).convert("RGB")
        origin = gt.convert("P")   
        origin = torchvision.transforms.functional.to_tensor(origin) - 0.5
        with torch.no_grad():
            origin = torch.stack([origin])
            origin = origin.to(device)
            outs = unet(origin)
            # outs = outs.float()
            # out = (outs[:,0]+outs[:,1])/2
            out = outs.float()
            pil_origin = torchvision.transforms.functional.to_pil_image(origin[0] + 0.5).convert("RGB")
            pil_mask_pred = torchvision.transforms.functional.to_pil_image(out[0] + 0.5).convert("RGB")
            # pil_mask_pred = pil_mask_pred.resize((2828,2320)) #(2828,2320)
            
            pil_mask_pred = pil_mask_pred.convert("L")
            
            image_array = np.array(pil_mask_pred)

            white_mask = np.where(image_array >= 240)
            image_array[white_mask] = 0
            pil_mask_pred = Image.fromarray(image_array)
            
           
            # 进行二值化处理
            pil_mask = pil_mask_pred.point(lambda p: p > 35 and 255)
            
            
            pil_mask_pred.save(images_folder_pred / img_item)
            pil_origin.save(images_folder_gt / img_item)
            pil_mask.save(images_folder_mask / img_item)
            # gt.save(images_folder_gt / img_item)
            
            
            plt.subplot(1, 3, 1)
            plt.title("origin image")
            plt.imshow(np.array(pil_origin))

            plt.subplot(1, 3, 2)
            plt.title("manually location_map_pred")
            plt.imshow(np.array(pil_mask_pred))
            
            plt.subplot(1, 3, 3)
            plt.title("manually mask_pred")
            plt.imshow(np.array(pil_mask))
            
            
            # plt.savefig(images_folder_mix / img_item, bbox_inches='tight')

        print(".", end="")                
                           
def val_real_chestdet():
    # unet = PretrainedUNet(1, 2, True, "bilinear")
    
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = 1
    config_vit.n_skip = 3
    # if args.vit_name.find('R50') != -1:
    config_vit.patches.grid = (int(512 / 16), int(512 / 16))
    unet = ViT_seg(config_vit, img_size=512, num_classes=config_vit.n_classes).cuda()
    model_name = "unet-6v-latest.pt"
    unet.load_state_dict(torch.load(images_folder / model_name, map_location=torch.device("cpu")))
    unet.to(device)
    unet.eval();


    img_root = "paired_data/Chest-Det/test_data"
    json_root = "paired_data/Chest-Det/test.json"
    with open(json_root, 'r') as f:
        dict = json.load(f)
    dataset_id = "chestxdet_test"
    images_folder_gt = Path("images_transunet_"+add,dataset_id,"gt")
    images_folder_img = Path("images_transunet_"+add,dataset_id,"img")
    images_folder_pred = Path("images_transunet_"+add,dataset_id,"pred")
    images_folder_mask = Path("images_transunet_"+add,dataset_id,"mask")
    # images_folder_mix = Path("images_transunet_"+add,dataset_id,"mix")
    if not os.path.exists(images_folder_gt):
        os.makedirs(images_folder_gt)
    if not os.path.exists(images_folder_img):
        os.makedirs(images_folder_img)
    if not os.path.exists(images_folder_pred):
        os.makedirs(images_folder_pred)
    if not os.path.exists(images_folder_mask):
        os.makedirs(images_folder_mask)
    # if not os.path.exists(images_folder_mix):
    #     os.makedirs(images_folder_mix)
        
    # img_list = os.listdir(img_root)
    # img_list.sort()
    for i,info in enumerate(dict[501:]):
        img_item = info['file_name'].replace('png','jpg')
        labels = "_".join(info['syms'])
        bbox = info['boxes']
        bbox_new = []
        for line in bbox:
            line_new = []
            for j,item in enumerate(line):
                # if j < 2:
                #     line_new.append(item)
                # else:
                line_new.append(round(item/2))
            bbox_new.append(line_new)  
        
        img_dir = os.path.join(img_root,img_item)
        img_item = img_item.replace(".jpg","_"+labels+".jpg")
        # img_out = os.path.join(images_folder_pred,img_item)
        # origin = Image.open(img_dir).convert("P")
        # gt1 = Image.open(img_dir).convert("RGB")
        gt = Image.open(img_dir).resize((512, 512), Image.ANTIALIAS).convert("RGB")
        origin = gt.convert("P")
        draw = ImageDraw.Draw(gt) # 在上面画画
        if bbox != []:
            for bbox_item in bbox_new:
                draw.rectangle(bbox_item, outline=(255,0,0))
        # gt1 = gt1.resize((512, 512), Image.ANTIALIAS)
        
        origin = torchvision.transforms.functional.to_tensor(origin) - 0.5
        with torch.no_grad():
            origin = torch.stack([origin])
            origin = origin.to(device)
            outs = unet(origin)
            # outs = outs.float()
            # out = (outs[:,0]+outs[:,1])/2
            out = outs.float()
            pil_origin = torchvision.transforms.functional.to_pil_image(origin[0] + 0.5).convert("RGB")
            pil_mask_pred = torchvision.transforms.functional.to_pil_image(out[0] + 0.5).convert("RGB")
            # pil_mask_pred = pil_mask_pred.resize((2828,2320)) #(2828,2320)
            
            pil_mask_pred = pil_mask_pred.convert("L")
            
            image_array = np.array(pil_mask_pred)

            white_mask = np.where(image_array >= 240)
            image_array[white_mask] = 0
            pil_mask_pred = Image.fromarray(image_array)
            
           


            # 进行二值化处理
            pil_mask = pil_mask_pred.point(lambda p: p > 45 and 255)
            
            
            pil_mask_pred.save(images_folder_pred / img_item)
            pil_origin.save(images_folder_img / img_item)
            pil_mask.save(images_folder_mask / img_item)
            gt.save(images_folder_gt / img_item)
            
            
            # plt.subplot(1, 3, 1)
            # plt.title("origin image")
            # plt.imshow(np.array(gt))

            # plt.subplot(1, 3, 2)
            # plt.title("manually location_map_pred")
            # plt.imshow(np.array(pil_mask_pred))
            
            # plt.subplot(1, 3, 3)
            # plt.title("manually mask_pred")
            # plt.imshow(np.array(pil_mask))
            
            
            # plt.savefig(images_folder_mix / img_item, bbox_inches='tight')

        print(".", end="")

def shorten_filename(filename, limit=180):
    """返回合适长度文件名，中间用...显示"""
    if len(filename) <= limit:
        return filename
    else:
        return filename[:int(limit / 2) - 3] + '...' + filename[len(filename) - int(limit / 2):]
   
def val_real_padchest():
    # unet = PretrainedUNet(1, 2, True, "bilinear")
    
    config_vit = CONFIGS_ViT_seg['R50-ViT-B_16']
    config_vit.n_classes = 1
    config_vit.n_skip = 3
    # if args.vit_name.find('R50') != -1:
    config_vit.patches.grid = (int(512 / 16), int(512 / 16))
    unet = ViT_seg(config_vit, img_size=512, num_classes=config_vit.n_classes).cuda()
    model_name = "unet-6v-latest.pt"
    unet.load_state_dict(torch.load(images_folder / model_name, map_location=torch.device("cpu")))
    unet.to(device)
    unet.eval();


    img_root = "paired_data/PadChest"
    csv_root = "PADCHEST_chest_x_ray_images_labels_160K_01.02.19.csv"
    
    # column_names = ["ImageID", "Labels"]
    dict = pd.read_csv(csv_root,low_memory=False)
    name_list = dict["ImageID"].tolist()
    label_list = dict["Labels"].tolist()

    dataset_id = "padchest_sampledata"
    images_folder_gt = Path("images_transunet_padchest_"+add,dataset_id,"gt")
    images_folder_img = Path("images_transunet_padchest_"+add,dataset_id,"img")
    images_folder_pred = Path("images_transunet_padchest_"+add,dataset_id,"pred")
    images_folder_mask = Path("images_transunet_padchest_"+add,dataset_id,"mask")
    # images_folder_mix = Path("images_transunet_padchest_"+add,dataset_id,"mix")
    if not os.path.exists(images_folder_gt):
        os.makedirs(images_folder_gt)
    if not os.path.exists(images_folder_img):
        os.makedirs(images_folder_img)
    if not os.path.exists(images_folder_pred):
        os.makedirs(images_folder_pred)
    if not os.path.exists(images_folder_mask):
        os.makedirs(images_folder_mask)
    # if not os.path.exists(images_folder_mix):
    #     os.makedirs(images_folder_mix)
        
    img_list = os.listdir(img_root)
    # img_list.sort()
    for i,img_item in enumerate(img_list):
        id = name_list.index(img_item.replace('jpg','png'))
        labels = label_list[id]
        img_dir = os.path.join(img_root,img_item)
        img_item = img_item.replace(".jpg","_"+labels+".jpg")
        img_item = shorten_filename(img_item)
        # img_out = os.path.join(images_folder_pred,img_item)
        # origin = Image.open(img_dir).convert("P")
        # gt1 = Image.open(img_dir).convert("RGB")
        gt = Image.open(img_dir).convert("RGB")
        origin = gt.convert("P")
        

        # gt1 = gt1.resize((512, 512), Image.ANTIALIAS)
        
        origin = torchvision.transforms.functional.to_tensor(origin) - 0.5
        with torch.no_grad():
            origin = torch.stack([origin])
            origin = origin.to(device)
            outs = unet(origin)
            # outs = outs.float()
            # out = (outs[:,0]+outs[:,1])/2
            out = outs.float()
            pil_origin = torchvision.transforms.functional.to_pil_image(origin[0] + 0.5).convert("RGB")
            pil_mask_pred = torchvision.transforms.functional.to_pil_image(out[0] + 0.5).convert("RGB")
            # pil_mask_pred = pil_mask_pred.resize((2828,2320)) #(2828,2320)
            
            pil_mask_pred = pil_mask_pred.convert("L")
            
            image_array = np.array(pil_mask_pred)

            white_mask = np.where(image_array >= 240)
            image_array[white_mask] = 0
            pil_mask_pred = Image.fromarray(image_array)
            
           


            # 进行二值化处理
            pil_mask = pil_mask_pred.point(lambda p: p > 30 and 255)
            
            
            pil_mask_pred.save(images_folder_pred / img_item)
            pil_origin.save(images_folder_img / img_item)
            pil_mask.save(images_folder_mask / img_item)
            gt.save(images_folder_gt / img_item)
            
            
            # plt.subplot(1, 3, 1)
            # plt.title("origin image")
            # plt.imshow(np.array(gt))

            # plt.subplot(1, 3, 2)
            # plt.title("manually location_map_pred")
            # plt.imshow(np.array(pil_mask_pred))
            
            # plt.subplot(1, 3, 3)
            # plt.title("manually mask_pred")
            # plt.imshow(np.array(pil_mask))
            
            
            # plt.savefig(images_folder_mix / img_item, bbox_inches='tight')

        print(".", end="")   
    

if __name__ == "__main__":
    datasets,dataloaders = load()
    # train(datasets,dataloaders)
    val(datasets,dataloaders)
    # val_real_chexpert()
    # val_real_chestdet()
    # val_real_padchest()
