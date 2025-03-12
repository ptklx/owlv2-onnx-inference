import time
import numpy as np
import cv2
import json
import onnxruntime as ort
import random
import math
import os
import sys
import functools
from typing import List, Optional

from transformers import Owlv2Processor  ##tmp


ROOT_PATH = os.getenv("ROOT_PATH", os.path.dirname(os.path.realpath(__file__)))
sys.path.append(ROOT_PATH)

import simple_tokenizer


current_p =os.path.dirname(os.path.abspath(__file__))

DEFAULT_BPE_PATH = os.path.join(current_p,"weights/bpe_simple_vocab_16e6.txt.gz")
DEFAULT_BPE_URL = 'https://github.com/openai/CLIP/blob/main/clip/bpe_simple_vocab_16e6.txt.gz?raw=true'
def tokenize(text: str, max_token_len: int = 77) -> List[int]:
    tokenizer = build_tokenizer()
    sot_token = tokenizer.encoder['<|startoftext|>']
    eot_token = tokenizer.encoder['<|endoftext|>']
    tokens = [sot_token] + tokenizer.encode(text) + [eot_token]
    output = [0] * max_token_len
    output[:min(max_token_len, len(tokens))] = tokens[:max_token_len]
    return output

@functools.lru_cache(maxsize=1)
def build_tokenizer(
    bpe_path: Optional[str] = DEFAULT_BPE_PATH
) -> simple_tokenizer.SimpleTokenizer:

  return simple_tokenizer.SimpleTokenizer(bpe_path)


######draw

def draw_detect_res(img, det_pred,local=True,beer_soda=False):
    '''
    检测结果绘制
    '''
    color = (0, 255, 255)
    color_step = 255
    object_name = '_'
    img = img.astype(np.uint8)
    for i in range(len(det_pred)):
        x1, y1, x2, y2 = [int(t) for t in det_pred[i][:4]]
        score = det_pred[i][4]
        cls_id = int(det_pred[i][5])
        # print(i + 1, [x1, y1, x2, y2], score, coco_tmp[cls_id])
        cv2.putText(img, str(np.round(score, 3)), (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        if local==False: # 为了区别颜色
            cls_id+=1
            cv2.rectangle(img, (x1, y1), (x2 , y2 ), (0, int(cls_id * color_step), int(255 - cls_id * color_step)),
                        thickness=1)
        else:
            cv2.rectangle(img, (x1, y1), (x2 , y2 ), (0, int(cls_id * color_step), int(255 - cls_id * color_step)),
                        thickness=2)

    return img

def draw_circle_res(img,point,name="null"):
    radius=6
    color = (0,0,255)
    thickness=-1
  # cv2.putText(img, name, (int(point[0]), int(point[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    cv2.circle(img, (int((point[0]+point[2])//2),int((point[1]+point[3])//2)), radius, color, thickness)
    return img

##############

def customized_filtration_min_max(img, det_pred):
    middle_w = img.shape[1]*0.75   # 宽，
    middle_h = img.shape[0]*0.75   # 高
    det_pred_result = []
    for i in range(len(det_pred)):
        x1, y1, x2, y2 = [int(t) for t in det_pred[i][:4]]
        x,y,w,h = x1,y1,x2-x1,y2-y1
        if w>middle_w or h >middle_h:  # 排除过大的框
           continue
        if  w<middle_w/20 or h <middle_h/25: # 排除过小的框
            continue
        # if x <0 or x>img.shape[1]-1:
        #    continue
        if x <0 :
            det_pred[i][0]=0
        if x>img.shape[1]-1:
           det_pred[i][0]=img.shape[1]-1
        if y <0:
            det_pred[i][1]=0
        if y>img.shape[0]-1:
            det_pred[i][1]=img.shape[0]-1
        det_pred_result.append(det_pred[i])
        
    return det_pred_result

def get_max_score(det_pred):
    select_box = sorted(det_pred, key=lambda va: va[4],reverse=True)  # 从大到小
    return select_box[0]


###########pcl


def pcl_lenghs(matrix,center=[2,2],radius=2):
    # 指定中心点和半径
    # center = (2, 2)  # 中心点 (行, 列)
    # radius = 1
    # 获取矩阵的形状
    rows, cols = matrix.shape
    # 计算在半径内的点
    mean_values = []

    # 加速
    tem_radius = radius+1
    r_start = center[0]-tem_radius if (center[0]-tem_radius)>0 else 0
    r_end = center[0]+tem_radius if (center[0]+tem_radius)< rows else rows
    c_start = center[1]-tem_radius if (center[1]-tem_radius)>0 else 0
    c_end = center[1]+tem_radius if (center[1]+tem_radius)>0 else cols

    for i in range(r_start,r_end):
        for j in range(c_start,c_end):
            # 计算到中心点的欧几里得距离
            if (i - center[0])**2 + (j - center[1])**2 <= radius**2:
                mean_values.append(matrix[i, j])
    if mean_values==[]:
        return 0
    # 计算均值
    mean_value = np.mean(mean_values)
    return  int(mean_value)



def pcl_max_lenghs(matrix,point):
    cx,cy,h,w = point
    roct = matrix[int(cy-h /10 ):int(cy + h / 10 ), int(cx - w / 10):int(cx + w / 2 )]
    mean_v= np.max(roct)
    return mean_v

def get_mid_pos(box,depth_data,randnum=250,max_value=3000):
    distance_list = []
    # mid_pos = [(box[0] + box[2])//2, (box[1] + box[3])//2] #确定索引深度的中心像素位置
    mid_pos = [(box[0] + box[2])//2, box[1]+ (box[3] - box[1])//3]
    # mid_pos = [box[0],box[1]]
    h,w = depth_data.shape
    min_val = min(abs(box[2] - box[0]), abs(box[3] - box[1])-4) #确定深度搜索范围
    # min_val=min(abs(box[2]-4),abs(box[3]-4))
    #print(box,)
    for i in range(randnum):
        # bias_1 = random.randint(-min_val//2, min_val//2)
        # bias_2 = random.randint(-min_val//2, min_val//2)
        bias = random.randint(-min_val//2, min_val//2)
        y_p = int(mid_pos[1] + bias)
        x_p = int(mid_pos[0] + bias)
        y_p = y_p if y_p>0 else 0
        y_p = h-1 if y_p>h-1 else y_p

        x_p = x_p if x_p>0 else 0
        x_p = w-1 if x_p>w-1 else x_p
        dist = depth_data[y_p, x_p]
        # dist = depth_data[int(abs(mid_pos[1] + bias_1)), int(abs(mid_pos[0] + bias_2))]
        # cv2.circle(frame, (int(mid_pos[0] + bias), int(mid_pos[1] + bias)), 4, (255,0,0), -1)
        #print(int(mid_pos[1] + bias), int(mid_pos[0] + bias))
        if dist:
            distance_list.append(dist)
    if distance_list==[]:
        return 100000
    distance_list = np.array(distance_list)
    distance_sort = np.sort(distance_list)
    distance_list= distance_sort[randnum//2-randnum//3:randnum//2+randnum//6] #冒泡排序+中值滤波

    distance_list = [value for value in distance_list if value <= max_value]
    if distance_list==[]:
        return 100000
    #print(distance_list, np.mean(distance_list))
    dis_mean = np.mean(distance_list)
    print("dis_mean",dis_mean)
    if math.isnan(dis_mean):
        return 100000
    else:
        return int(dis_mean)



########################owlvit det start
# 将中心格式转换为角格式
def center_to_corners_format(bboxes_center):
    center_x, center_y, width, height = bboxes_center[..., 0], bboxes_center[..., 1], bboxes_center[..., 2], bboxes_center[..., 3]
    bbox_corners = np.stack(
        # top left x, top left y, bottom right x, bottom right y
        [center_x - 0.5 * width, center_y - 0.5 * height, center_x + 0.5 * width, center_y + 0.5 * height],
        axis=-1
    )
    return bbox_corners

# def center_to_corners_format(bboxes_center: np.ndarray) -> np.ndarray:
#     center_x, center_y, width, height = bboxes_center.T
#     bboxes_corners = np.stack(
#         # top left x, top left y, bottom right x, bottom right y
#         [center_x - 0.5 * width, center_y - 0.5 * height, center_x + 0.5 * width, center_y + 0.5 * height],
#         axis=-1,
#     )
#     return bboxes_corners

# 后处理方法
def post_process_object(logits, boxes, target_sizes):
    logits = np.array(logits)
    boxes = np.array(boxes)
    # 对 logits 进行最大值和索引的处理
    probs = np.max(logits, axis=-1)
    scores = 1 / (1 + np.exp(-probs))  # Sigmoid 处理
    labels = np.argmax(logits, axis=-1)
    # 转换为 [x0, y0, x1, y1] 格式
    boxes = center_to_corners_format(boxes)
    # 从相对坐标 [0, 1] 转换为绝对坐标 [0, height] 和 [0, width]
    img_h, img_w = target_sizes[:, 0], target_sizes[:, 1]
    max_size= max(img_h, img_w)                     #############################################################
    # scale_fct = np.stack([img_w, img_h, img_w, img_h], axis=1)  # 计算缩放因子  owlvit
    scale_fct = np.stack([max_size, max_size, max_size, max_size], axis=1)  # 计算缩放因子 owlv2
    boxes = boxes * scale_fct[:, None, :]
    # 创建结果字典
    # results = [{"scores": s, "labels": l, "boxes": b} for s, l, b in zip(scores, labels, boxes)]
    # return results
    return scores[0], labels[0], boxes[0]

def NMS(dets, scores, thresh):
    '''
    单类NMS算法
    dets.shape = (N, 5), (left_top x, left_top y, right_bottom x, right_bottom y, Scores)
    '''
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    areas = (y2 - y1 + 1) * (x2 - x1 + 1)
    keep = []
    index = scores.argsort()[::-1]
    while index.size > 0:
        i = index[0]  # every time the first is the biggst, and add it directly
        keep.append(i)
        x11 = np.maximum(x1[i], x1[index[1:]])  # calculate the points of overlap
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])
        w = np.maximum(0, x22 - x11 + 1)  # the weights of overlap
        h = np.maximum(0, y22 - y11 + 1)  # the height of overlap
        overlaps = w * h
        ious = overlaps / (areas[i] + areas[index[1:]] - overlaps)
        idx = np.where(ious <= thresh)[0]
        index = index[idx + 1]  # because index start from 1

    return keep

def resize_with_padding(image, target_width, target_height):
    # 获取原始图像的宽度和高度
    (original_height, original_width) = image.shape[:2]

    # 计算宽高比
    aspect_ratio = original_width / original_height
    target_aspect_ratio = target_width / target_height

    # 根据目标尺寸调整缩放比例
    if aspect_ratio > target_aspect_ratio:
        # 固定宽度，调整高度
        new_width = target_width
        new_height = int(target_width / aspect_ratio)
    else:
        # 固定高度，调整宽度
        new_height = target_height
        new_width = int(target_height * aspect_ratio)

    # 缩放图像
    resized_image = cv2.resize(image, (new_width, new_height))

    # 创建目标尺寸的图像，并填充0
    padded_image = np.zeros((target_height, target_width, 3), dtype=np.uint8)

    # 计算填充位置
    start_x = (target_width - new_width) // 2
    start_y =0 # (target_height - new_height) // 2

    # 将缩放后的图像放入填充图像的中心
    padded_image[start_y:start_y + new_height, start_x:start_x + new_width] = resized_image

    return padded_image

class owlvitOnnx:
    def __init__(self,path):
        super().__init__()
        self.yolo_session = ort.InferenceSession(path, providers=['CPUExecutionProvider',"CUDAExecutionProvider"])  #CUDAExecutionProvider
        self.input_names = [input.name for input in self.yolo_session.get_inputs()]

    def __call__(self, inputs):
        onnx_inputs = {}
        for idx,input in enumerate(inputs):
            onnx_inputs.update({self.input_names[idx]: input})
            
        output = self.yolo_session.run(None, onnx_inputs)
        return output
    
class owlvitDet:
    def __init__(self,path_image,path_text,path_post,score_threshold=0.1):
        self.session_image = owlvitOnnx(path_image)
        self.session_text = owlvitOnnx(path_text)
        self.session_post = owlvitOnnx(path_post)
        self.tokenizer = build_tokenizer()
        self.score_threshold = score_threshold ######
        self.iou_threshold = 0.3  #######
        self.input_w=960
        self.input_h = 960
        ####: tmp
        self.processor = Owlv2Processor.from_pretrained("owlv2-base-patch16-ensemble")
  
    def preprocess_input(self,img,text_queries):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转为 RGB 格式  
        # 图像大小调整
        # img = cv2.resize(img, (self.input_w,self.input_h))
        img = resize_with_padding(img, self.input_w,self.input_h)

        # 转换为浮动类型并归一化 (标准化到 [0, 1])
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.48145466, 0.4578275, 0.40821073])  # ImageNet 的平均值
        std = np.array([0.26862954, 0.26130258, 0.27577711])   # ImageNet 的标准差
        img = (img - mean) / std
        # 将图像转换为 (C, H, W) 格式
        # img = np.transpose(img, (2, 0, 1))  # 变换维度顺序为 (channels, height, width)
        pixel_values = np.expand_dims(img, axis=0).astype(np.float32)  # 增加一个 batch 维度，(1, C, H, W)
        input_ids = np.array([self.tokenizer.encode(text_queries)  ]).reshape(-1)
        input_ids = np.pad([49406,*input_ids,49407],(0,16-len(input_ids)-2))
        input_ids =np.expand_dims(input_ids, axis=0)
        attention_mask = (input_ids > 0).astype(np.int64)
        return  input_ids, attention_mask,pixel_values
    
    def post_process(self,img,logits,pred_boxes,score_threshold=0.04):
        result_out =[]
        target_sizes = np.array([img.shape[:2]])  # shape: (1, 2), height and width
        scores, labels, boxes = post_process_object(logits=logits,boxes=pred_boxes, target_sizes=target_sizes)
        # boxes, scores, labels = results[0]["boxes"], results[0]["scores"], results[0]["labels"]
        # font = cv2.FONT_HERSHEY_SIMPLEX
        iou_thred = self.iou_threshold###############
        indexs = NMS(boxes,scores,iou_thred)

        for ind in indexs:
        # for box, score, label in zip(boxes, scores, labels):
            box, score, label = boxes[ind], scores[ind], labels[ind]
            box = [int(i) for i in box.tolist()]
            # print('score >= score_threshold',score, score_threshold)
            if score >= score_threshold:
                box.append(score)
                box.append(label)
                result_out.append(box)
                # img = cv2.rectangle(img, box[:2], box[2:], (0,0,255), 1)
                # if box[3] + 25 > 768:
                #     y = box[3] - 8
                # else:
                #     y = box[3] + 15
                    
                # img = cv2.putText(
                #     img, text_input[label]+" "+str(np.round(score, 3)), (box[0], y), font, 0.5, (0,0,255), 1, cv2.LINE_AA
                # )
        return result_out

    def __call__(self,image,text_prompt):
        t1=time.time()
        input_ids,attention_mask,pixel_values=self.preprocess_input(image,text_prompt)
        pixel_values = pixel_values.transpose(0,3,1,2)
        # inputs = self.processor(text=text_prompt, images=image, return_tensors="pt")
        # input_ids,attention_mask ,pixel_values=inputs.data["input_ids"].numpy(),inputs.data["attention_mask"].numpy(),inputs.data["pixel_values"].numpy()
        input_ids=input_ids.astype(np.int64)
        # 图像推理
        
        out_features = self.session_image([pixel_values])
        image_features,pred_boxes= out_features[0],out_features[1]
        image_features = image_features.reshape(image_features.shape[0],-1,image_features.shape[-1])
        # image_features =np.array(output[0])
        # pred_boxes_=np.array(output[1])
        t2=time.time()
        # 文本推理
        text_features = self.session_text([attention_mask,input_ids])
        text_feature = text_features[0]
        # 后处理
        logits_= self.session_post([image_features,text_feature,attention_mask])
        t3=time.time()
        logits=np.array(logits_[0])      
        box_result=self.post_process(image,logits,pred_boxes,self.score_threshold)

        cost_time = (time.time()-t1)*1000
        # invoke_time.append(cost_time)
        print("=======================================")
        print(f"inference time {cost_time} ms , image cost time: {(t2-t1)*1000} ms, text post cost time: {(t3-t2)*1000} ms")
        print("=======================================")
        return  box_result


# import sys
# if sys.platform.startswith('win'):
def main_camera_pcl():
    import pyrealsense2 as rs
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    # config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    # config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
    align_m=True
    if align_m:
        # 0:彩色图像对齐到深度图;
        # 1:深度图对齐到彩色图像
        ALIGN_WAY = 1
        # Start streaming
        # 设置对齐方式
        align_to = rs.stream.color if ALIGN_WAY == 1 else rs.stream.depth
        align = rs.align(align_to)

    pipeline.start(config)

    path_image = os.path.join(current_p,"models/owlvit-image-infer.quant.onnx")
    path_text = os.path.join(current_p,"models/owlvit-text.onnx")
    path_post = os.path.join(current_p,"models/owlvit-post.onnx")
    qnn_owlvit = owlvitDet(path_image,path_text,path_post)

    text_prompt = "a photo of a bottle"

    while True:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        if align_m:
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
        else:
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue
        # Convert images to numpy arrays
        dframe = np.asanyarray(depth_frame.get_data())
        frame = np.asanyarray(color_frame.get_data())

        preds=qnn_owlvit(frame,text_prompt)
        if len(preds):
            det_pred = customized_filtration_min_max(frame, preds)   # 删除不在范围中的

            res_img = draw_detect_res(frame, det_pred) #
            one_point = get_max_score(det_pred)

            circle_frame = draw_circle_res(res_img,one_point)

            the_max_distance = 3000
            mean_v =get_mid_pos(one_point,dframe,200,max_value=the_max_distance)
            if mean_v>the_max_distance:
                # class_names = [ 'soda', 'beer', 'water','coke', 'fanta','test']
                print("mean_v:%f"%mean_v)
                # cv2.imwrite("box_result.png", showframe)
            cv2.imshow("test",circle_frame)
            cv2.waitKey(1)





def main():
    imgs = os.path.join(current_p,"bottle.jpg")
    print("Start main ... ...")
    path_image = os.path.join(current_p,"weights/owlv2-image.onnx")
    path_text = os.path.join(current_p,"weights/owlv2-text.onnx")
    path_post = os.path.join(current_p,"weights/owlv2-post.onnx")
    qnn_owlvit = owlvitDet(path_image,path_text,path_post)
    # text_prompt = "a photo of a cap"
    text_prompt = "detect plastic bottle"
    frame = cv2.imread(imgs)
    preds=qnn_owlvit(frame,text_prompt)

    if len(preds):
        det_pred = customized_filtration_min_max(frame, preds)   # 删除不在范围中的
        res_img = draw_detect_res(frame, det_pred) #        
        cv2.imshow("test",res_img)
        cv2.waitKey(0)
        # save_path=os.path.join(current_p,"result.jpg")
        # cv2.imwrite(save_path, res_img)   
       
    return True

def main_camera():
    # imgs = os.path.join(current_p,"bottle.jpg")
    print("Start main ... ...")
    # path_image = os.path.join(current_p,"weights_960/owlv2-image.onnx")
    path_image = os.path.join(current_p,"weights_960/owlv2-image_int8.onnx")

    path_text = os.path.join(current_p,"weights_960/owlv2-text.onnx")
    path_post = os.path.join(current_p,"weights_960/owlv2-post.onnx")
    qnn_owlvit = owlvitDet(path_image,path_text,path_post)
    text_prompt = "a photo of a  keyboard"
    # text_prompt = "detect plastic bottle"
    # frame = cv2.imread(imgs)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("无法打开摄像头")
        exit()

    while True:
        # 读取一帧
        ret, frame = cap.read()
        # 如果读取成功，ret 为 True
        if not ret:
            print("无法读取帧 (流结束?). 退出...")
            break
        preds=qnn_owlvit(frame,text_prompt)

        if len(preds):
            det_pred = customized_filtration_min_max(frame, preds)   # 删除不在范围中的
            res_img = draw_detect_res(frame, det_pred) #        
            cv2.imshow("test",res_img)
        key = cv2.waitKey(1)

        if key == 27:
            text_prompt = input("请输入目标： ")
            print(text_prompt)
            # save_path=os.path.join(current_p,"result.jpg")
            # cv2.imwrite(save_path, res_img)   
        
    return True



if __name__ == "__main__":
    # if sys.platform.startswith('win'):
        # main_camera_pcl()
    # else:
    # main()
    main_camera()

