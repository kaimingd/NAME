import os 
import cv2
import numpy as np

ref_path = 'paired_data/adapter_chestxray_reference'
exmaple_path = 'paired_data/adapter_chestxray_paired'
out_path = 'paired_data/adapter_chestxray_paired_minus1'

# labels = ["consolidation","enlargedcardiomediastinum"]# "cardiomegaly","consolidation","edema","enlargedcardiomediastinum","fracture","lunglesion",
#            # "lungopacity","pleuraleffusion","pleuralother","nofinding","pneumothorax","pneumonia","supportdevices"]
# labels = ["pneumonia","supportdevices"]
# label = ["atelectasis","consolidation","enlargedcardiomediastinum","fracture","lunglesion",
#             "lungopacity","pleuralother","pneumothorax","pneumonia","supportdevices"]
labels = ["atelectasis", "cardiomegaly","consolidation","edema","enlargedcardiomediastinum","fracture","lunglesion",
            "lungopacity", "pleuraleffusion","pleuralother","nofinding","pneumothorax","pneumonia","supportdevices"]

# labels = ["normal","unchanged","exclude","interstitialpattern","pleuraleffusion","pneumonia","alveolarpattern","infiltrates","chronicchanges","increaseddensity","laminaratelectasis","costophrenicangleblunting","pseudonodule","fibroticband","nodule","atelectasis","COPDsigns","cardiomegaly","heartinsufficiency","apicalpleuralthickening","vertebraldegenerativechanges","volumeloss","scoliosis","callusribfracture","calcifiedgranuloma","suboptimalstudy","bronchiectasis","airtrapping","vascularhilarenlargement","vertebralanteriorcompression","kyphosis","bronchovascularmarkings","consolidation","hilarcongestion","pacemaker","nippleshadow","aorticelongation","suturematerial","metal","calcifieddensities","hilarenlargement","goiter","hemidiaphragmelevation","pulmonarymass","NSGtube","emphysema","aorticatheromatosis","endotrachealtube","granuloma","trachealshift","centralvenouscatheterviajugularvein","pleuralthickening","bullas","superiormediastinalenlargement","osteosynthesismaterial","tuberculosissequelae","hiatalhernia","flatteneddiaphragm","lobaratelectasis","scleroticbonelesion","centralvenouscatheterviasubclavianvein","hypoexpansion","pulmonaryfibrosis","pulmonaryedema","ribfracture","vascularredistribution","groundglasspattern","supraaorticelongation","mediastiniclipomatosis","adenopathy","pneumothorax","centralvenouscatheter","cavitation","calcifiedadenopathy","chestdraintube","multiplenodules","mediastinalenlargement","dualchamberdevice","diaphragmaticeventration","non-axialarticulardegenerativechanges","lungmetastasis","humeralfracture","reservoircentralvenouscatheter","osteopenia","dai","sternotomy","calcifiedpleuralthickening","vertebralcompression","aorticbuttonenlargement","mediastinalmass","reticularinterstitialpattern","surgery","vertebralfracture","hyperinflatedlung","claviclefracture","osteoporosis","minorfissurethickening","axialhyperostosis","bonemetastasis","endonvessel","surgerybreast","thoraciccagedeformation","mastectomy","tuberculosis","reticulonodularinterstitialpattern","subcutaneousemphysema","atypicalpneumonia","centralvenouscatheterviaumbilicalvein","loculatedpleuraleffusion","surgeryneck","miliaryopacities","abnormalforeignbody","ascendentaorticelongation","calcifiedpleuralplaques","airfluidlevel","softtissuemass","segmentalatelectasis","postradiotherapychanges","pulmonaryarteryenlargement","descendentaorticelongation","fissurethickening","subacromialspacenarrowing","pericardialeffusion","mediastinalshift","pectumexcavatum","lyticbonelesion","tracheostomytube","prosthesis","heartvalvecalcified","aorticaneurysm","calcifiedfibroadenoma","azygoesophagealrecessshift","hydropneumothorax","endoprosthesis","asbestosissigns","surgeryheart","respiratorydistress","cyst","esophagicdilatation","pleuralmass","empyema","nofinding"]


if not os.path.exists(out_path):
    os.makedirs(out_path)


# img_img_minus1
ref_list = os.listdir(ref_path)
ref_list = [ref for ref in ref_list if ref[-7:-4]=='ref']
ref_list.sort()
exm_list = os.listdir(exmaple_path)
exm_list.sort()

for i,ref in enumerate(ref_list):
    ref_img_path = os.path.join(ref_path,ref)
    ref_mask_path = ref_img_path.replace('_ref.jpg','_seg.jpg')
    # ref_img =cv2.imread(ref_img_path).astype('float32') / 255
    ref_mask =cv2.imread(ref_mask_path).astype('float32') / 255
    for label in labels:
        exm_img_path2 = ref.replace('reference',label)
        exm_img_path1 = exm_img_path2.replace('_ref.jpg','.jpg')
        exm_img_path = os.path.join(exmaple_path,exm_img_path1)
        
        ref_img_path2 = ref.replace('reference','nofinding')
        ref_img_path1 = ref_img_path2.replace('_ref.jpg','.jpg')
        ref_img_path0 = os.path.join(exmaple_path,ref_img_path1)
        ref_img = cv2.imread(ref_img_path0).astype('float32') / 255


        exm_img = cv2.imread(exm_img_path).astype('float32') / 255
        del_img = (abs(ref_img-exm_img))**1
        del_img = ((ref_mask*del_img)*255).astype('uint8')
        # del_img = (((abs(ref_mask*del_img))**1.5)*255).astype('uint8')
        out_img_path = os.path.join(out_path,exm_img_path1)
        cv2.imwrite(out_img_path,del_img)
       
# img_img_minus1_binary  
# ref_list = os.listdir(ref_path)
# ref_list = [ref for ref in ref_list if ref[-7:-4]=='ref']
# ref_list.sort()
# exm_list = os.listdir(exmaple_path)
# exm_list.sort()

# for i,ref in enumerate(ref_list):
#     ref_img_path = os.path.join(ref_path,ref)
#     ref_mask_path = ref_img_path.replace('_ref.jpg','_seg.jpg')
#     # ref_img =cv2.imread(ref_img_path).astype('float32') / 255
#     ref_mask =cv2.imread(ref_mask_path).astype('float32') / 255
#     for label in labels:
#         exm_img_path2 = ref.replace('reference',label)
#         exm_img_path1 = exm_img_path2.replace('_ref.jpg','.jpg')
#         exm_img_path = os.path.join(exmaple_path,exm_img_path1)
        
#         ref_img_path2 = ref.replace('reference','nofinding')
#         ref_img_path1 = ref_img_path2.replace('_ref.jpg','.jpg')
#         ref_img_path0 = os.path.join(exmaple_path,ref_img_path1)
#         ref_img = cv2.imread(ref_img_path0).astype('float32') / 255
        
#         exm_img = cv2.imread(exm_img_path).astype('float32') / 255
#         del_img = (abs(ref_img-exm_img))**1
#         del_img = ((ref_mask*del_img)*255).astype('uint8')
#         gray = cv2.cvtColor(del_img,cv2.COLOR_BGR2GRAY)
#         del_img= cv2.threshold(gray,60,255,cv2.THRESH_BINARY)  #60   对不同的病取的阈值不同，需要斟酌
#         # del_img = (((abs(ref_img*exm_img))**2)*255).astype('uint8')
#         out_img_path = os.path.join(out_path,exm_img_path1)
#         cv2.imwrite(out_img_path,del_img[1])

        
# img_mask_minus
# ref_list = os.listdir(ref_path)
# ref_list = [ref for ref in ref_list if ref[-7:-4]=='seg']
# ref_list.sort()
# exm_list = os.listdir(exmaple_path)
# exm_list.sort()

# for i,ref in enumerate(ref_list):
#     ref_img_path = os.path.join(ref_path,ref)
#     ref_img =1 - cv2.imread(ref_img_path).astype('float32') / 255
#     for label in labels:
#         exm_img_path2 = ref.replace('reference',label)
#         exm_img_path1 = exm_img_path2.replace('_seg.jpg','.jpg')
#         exm_img_path = os.path.join(exmaple_path,exm_img_path1)
#         exm_img = cv2.imread(exm_img_path).astype('float32') / 255
#         del_img = ((abs(ref_img-exm_img))*255).astype('uint8')
#         # del_img = (((abs(ref_img*exm_img))**2)*255).astype('uint8')
#         out_img_path = os.path.join(out_path,exm_img_path1)
#         cv2.imwrite(out_img_path,del_img)
        
        
# img_mask_mul2
# ref_list = os.listdir(ref_path)
# ref_list = [ref for ref in ref_list if ref[-7:-4]=='seg']
# ref_list.sort()
# exm_list = os.listdir(exmaple_path)
# exm_list.sort()

# for i,ref in enumerate(ref_list):
#     ref_img_path = os.path.join(ref_path,ref)
#     ref_img =cv2.imread(ref_img_path).astype('float32') / 255
#     for label in labels:
#         exm_img_path2 = ref.replace('reference',label)
#         exm_img_path1 = exm_img_path2.replace('_seg.jpg','.jpg')
#         exm_img_path = os.path.join(exmaple_path,exm_img_path1)
#         exm_img = cv2.imread(exm_img_path).astype('float32') / 255
#         # del_img = ((abs(ref_img-exm_img))*255).astype('uint8')
#         del_img = (((abs(ref_img*exm_img))**2)*255).astype('uint8')
#         out_img_path = os.path.join(out_path,exm_img_path1)
#         cv2.imwrite(out_img_path,del_img)

# img_mask_mul1_binary
# ref_list = os.listdir(ref_path)
# ref_list = [ref for ref in ref_list if ref[-7:-4]=='seg']
# ref_list.sort()
# exm_list = os.listdir(exmaple_path)
# exm_list.sort()

# for i,ref in enumerate(ref_list):
#     ref_img_path = os.path.join(ref_path,ref)
#     ref_img =cv2.imread(ref_img_path).astype('float32') / 255
#     for label in labels:
#         exm_img_path2 = ref.replace('reference',label)
#         exm_img_path1 = exm_img_path2.replace('_seg.jpg','.jpg')
#         exm_img_path = os.path.join(exmaple_path,exm_img_path1)
#         exm_img = cv2.imread(exm_img_path).astype('float32') / 255
#         # del_img = ((abs(ref_img-exm_img))*255).astype('uint8')
#         del_img = (((abs(ref_img*exm_img))**1)*255).astype('uint8')
#         gray = cv2.cvtColor(del_img,cv2.COLOR_BGR2GRAY)
#         del_img= cv2.threshold(gray,160,255,cv2.THRESH_BINARY)  #160  80  对不同的病取的阈值不同，需要斟酌
#         out_img_path = os.path.join(out_path,exm_img_path1)
#         cv2.imwrite(out_img_path,del_img[1])
    
