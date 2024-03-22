# from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random
import os
import time
from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.hed import HEDdetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler


apply_hed = HEDdetector()

model = create_model('./mimic_models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./mimic_models/epoch=3-step=40387.ckpt', location='cuda'),strict=False)
model = model.cuda()
ddim_sampler = DDIMSampler(model)
# CUDA_VISIBLE_DEVICES=4 python inference_hed2image.py

label = ["atelectasis", "cardiomegaly","consolidation","edema","enlargedcardiomediastinum","fracture","lunglesion",
        "lungopacity", "pleuraleffusion","pleuralother","nofinding","pneumothorax","pneumonia","supportdevices"]
# label = ["normal","unchanged","exclude","interstitialpattern","pleuraleffusion","pneumonia","alveolarpattern","infiltrates","chronicchanges","increaseddensity","laminaratelectasis","costophrenicangleblunting","pseudonodule","fibroticband","nodule","atelectasis","COPDsigns","cardiomegaly","heartinsufficiency","apicalpleuralthickening","vertebraldegenerativechanges","volumeloss","scoliosis","callusribfracture","calcifiedgranuloma","suboptimalstudy","bronchiectasis","airtrapping","vascularhilarenlargement","vertebralanteriorcompression","kyphosis","bronchovascularmarkings","consolidation","hilarcongestion","pacemaker","nippleshadow","aorticelongation","suturematerial","metal","calcifieddensities","hilarenlargement","goiter","hemidiaphragmelevation","pulmonarymass","NSGtube","emphysema","aorticatheromatosis","endotrachealtube","granuloma","trachealshift","centralvenouscatheterviajugularvein","pleuralthickening","bullas","superiormediastinalenlargement","osteosynthesismaterial","tuberculosissequelae","hiatalhernia","flatteneddiaphragm","lobaratelectasis","scleroticbonelesion","centralvenouscatheterviasubclavianvein","hypoexpansion","pulmonaryfibrosis","pulmonaryedema","ribfracture","vascularredistribution","groundglasspattern","supraaorticelongation","mediastiniclipomatosis","adenopathy","pneumothorax","centralvenouscatheter","cavitation","calcifiedadenopathy","chestdraintube","multiplenodules","mediastinalenlargement","dualchamberdevice","diaphragmaticeventration","non-axialarticulardegenerativechanges","lungmetastasis","humeralfracture","reservoircentralvenouscatheter","osteopenia","dai","sternotomy","calcifiedpleuralthickening","vertebralcompression","aorticbuttonenlargement","mediastinalmass","reticularinterstitialpattern","surgery","vertebralfracture","hyperinflatedlung","claviclefracture","osteoporosis","minorfissurethickening","axialhyperostosis","bonemetastasis","endonvessel","surgerybreast","thoraciccagedeformation","mastectomy","tuberculosis","reticulonodularinterstitialpattern","subcutaneousemphysema","atypicalpneumonia","centralvenouscatheterviaumbilicalvein","loculatedpleuraleffusion","surgeryneck","miliaryopacities","abnormalforeignbody","ascendentaorticelongation","calcifiedpleuralplaques","airfluidlevel","softtissuemass","segmentalatelectasis","postradiotherapychanges","pulmonaryarteryenlargement","descendentaorticelongation","fissurethickening","subacromialspacenarrowing","pericardialeffusion","mediastinalshift","pectumexcavatum","lyticbonelesion","tracheostomytube","prosthesis","heartvalvecalcified","aorticaneurysm","calcifiedfibroadenoma","azygoesophagealrecessshift","hydropneumothorax","endoprosthesis","asbestosissigns","surgeryheart","respiratorydistress","cyst","esophagicdilatation","pleuralmass","empyema"]
# label = ["fracture","lunglesion","pleuralother","pneumothorax"]
# label = ["lunglesion"]
# strength_list = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
# label = []
# strength_list = [0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
# strength_list = [0.5,0.5,0.25,0.5,0.25,0.4,0.25,0.25,0.5,0.4,0.5,0.5,0.35,0.35]
strength_list = [0.35,0.5,0.35,0.5,0.25,0.3,0.35,0.25,0.5,0.3,0.5,0.5,0.35,0.35]
# strength_list = [0.3,0.35,0.3,0.5]
# strength_list = [0.40]
# label = ["atelectasis", "cardiomegaly","consolidation","edema","enlargedcardiomediastinum"]
# strength_list = [0.35,0.5,0.35,0.5,0.25]
# strength_list = [0.5,0.5,0.5,0.5,0.5]

# label = ["fracture","lunglesion","lungopacity", "pleuraleffusion","pleuralother",]
# strength_list = [0.3,0.35,0.25,0.5,0.3]
# strength_list = [0.5,0.5,0.5,0.5,0.5]

# label = ["nofinding","pneumothorax","pneumonia","supportdevices"]
# strength_list = [0.5,0.5,0.35,0.35]
# strength_list = [0.5,0.5,0.5,0.5]

out_dir = "paired_data/adapter_chestxray_paired"

if not os.path.exists(out_dir):
    os.makedirs(out_dir)

a_prompt = ""
n_prompt = ""   #增加负例可以更好的学习到
num_samples = 1
image_resolution = 512
detect_resolution = 512
ddim_steps = 150
guess_mode = False
scale = 9.0
# seed = 769374062
eta = 0
        
with torch.no_grad():
    for j in range(300):
        for i,label_item in enumerate(label):
            seed = 300000 + j
            cond_name = "paired_data/adapter_chestxray_reference/image_reference_"+str(seed)+"_ske.jpg"
            input_image = cv2.imread(cond_name)
            prompt = "a xray with {}".format(label_item)
            strength = strength_list[i]
            # strength = 0.5
            input_image = HWC3(input_image)
            detected_map = apply_hed(resize_image(input_image, detect_resolution))
            detected_map = HWC3(detected_map)
            img = resize_image(input_image, image_resolution)
            H, W, C = img.shape

            detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_LINEAR)

            control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
            control = torch.stack([control for _ in range(num_samples)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            if seed == -1:
                seed = random.randint(0, 65535)
            seed_everything(seed)

            if config.save_memory:
                model.low_vram_shift(is_diffusing=False)

            cond = {"c_concat": [control], "c_crossattn": [model.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
            un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model.get_learned_conditioning([n_prompt] * num_samples)]}
            shape = (4, H // 8, W // 8)

            if config.save_memory:
                model.low_vram_shift(is_diffusing=True)

            model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
            
            # start_time = time.time()
            samples, intermediates = ddim_sampler.sample(ddim_steps, num_samples,
                                                        shape, cond, verbose=False, eta=eta,
                                                        unconditional_guidance_scale=scale,
                                                        unconditional_conditioning=un_cond)

            if config.save_memory:
                model.low_vram_shift(is_diffusing=False)

            x_samples = model.decode_first_stage(samples)
            # end_time = time.time()
            # print("运行时间：", end_time - start_time, "秒")

            x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            img_dir = "image_"+label_item+"_"+str(seed)+".jpg"
            img_path = os.path.join(out_dir,img_dir)
            cv2.imwrite(img_path,x_samples[0])
            
            # img_minus_dir = os.path.join(out_dir, "image_"+label_item+"_"+str(seed)+"_min2.jpg")
            # indir1 = "/hhd/1/dkm/LCDG/work_dir_diffuser/sketch/adapter_chestxray_reference/image_reference_300200_ref.jpg"
            # indir2 = "/hhd/1/dkm/LCDG/work_dir_diffuser/sketch/adapter_chestxray_reference/image_reference_300200_seg.jpg"
            # ref_img = cv2.imread(indir1).astype('float32') / 255
            # ref_mask = cv2.imread(indir2).astype('float32') / 255
            # exm_img = x_samples[0].astype('float32') / 255
            # del_img = (abs(ref_img-exm_img))
            # del_img = ((ref_mask*del_img)*255).astype('uint8')
            # cv2.imwrite(img_minus_dir,del_img)
            
            # results = [x_samples[i] for i in range(num_samples)]

