# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Run YOLOv5 detection inference on images, videos, directories, globs, YouTube, webcam, streams, etc.

Usage - sources:
    $ python detect.py --weights yolov5s.pt --source 0                               # webcam
                                                     img.jpg                         # image
                                                     vid.mp4                         # video
                                                     path/                           # directory
                                                     'path/*.jpg'                    # glob
                                                     'https://youtu.be/Zgi9g1ksQHc'  # YouTube
                                                     'rtsp://example.com/media.mp4'  # RTSP, RTMP, HTTP stream

Usage - formats:
    $ python detect.py --weights yolov5s.pt                 # PyTorch
                                 yolov5s.torchscript        # TorchScript
                                 yolov5s.onnx               # ONNX Runtime or OpenCV DNN with --dnn
                                 yolov5s_openvino_model     # OpenVINO
                                 yolov5s.engine             # TensorRT
                                 yolov5s.mlmodel            # CoreML (macOS-only)
                                 yolov5s_saved_model        # TensorFlow SavedModel
                                 yolov5s.pb                 # TensorFlow GraphDef
                                 yolov5s.tflite             # TensorFlow Lite
                                 yolov5s_edgetpu.tflite     # TensorFlow Edge TPU
                                 yolov5s_paddle_model       # PaddlePaddle
"""

import argparse
import os
import platform
import sys
from pathlib import Path
import numpy as np

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.95,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=True,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=5,  # video frame-rate stride
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        view_img = check_imshow(warn=True)
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    with open(r'./res_info/1.txt', 'a+', encoding='utf-8') as test1:
        test1.truncate(0)
    with open(r'./res_info/4.txt', 'a+', encoding='utf-8') as test2:
        test2.truncate(0)

    cabinet1_list=[]
    cabinet4_list=[]
    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        #-------------1，4机柜判定flag初始化-----------------#
        count = 0
        flag1 = False
        flag2 = False
        #------------1，4机柜判定flag初始化结束----------------#


        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))

            # -------------初始化1，4柜指示灯记录数组matrix_01和matrix_04------------------------------#
            matrix_01 = [[i for i in range(2)] for i in range(2)]
            count_01 = 0
            matrix_04 = [[i for i in range(2)] for i in range(5)]
            count_04 = 0
            # -------------初始化1，4柜指示灯记录数组matrix_01和matrix_04结束------------------------------#

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                # -------------该帧中是否为1，4机柜判定------------------------------#
                for *xyxy, conf, cls in det:
                    count += 1
                    if int(cls) == 0 and float(conf) >= 0.95:
                        flag1 = True
                    if int(cls) == 1 and float(conf) >= 0.95:
                        flag2 = True
                # -------------该帧中是否为1，4机柜判定结束------------------------------#

                # -----------该帧是否为目标对象判定-----------------------------------#
                if (count == 3 and flag1) or (count == 6 and flag2):
                # -----------该帧是否为目标对象判定结束-----------------------------------#

                    # -----------目标对象判定数据读取(x坐标值、类别值open->2，close->3)-----------------------------------#
                    for *xyxy, conf, cls in reversed(det):
                        if count == 3 and int(cls) != 0:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            # line = (xywh[0], cls, conf) if save_conf else (xywh[0], cls)  # label format
                            matrix_01[count_01][0], matrix_01[count_01][1] = xywh[0], int(cls)
                            count_01 += 1
                        if count == 6 and int(cls) != 1:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            # line = (xywh[0], cls, conf) if save_conf else (xywh[0], cls)  # label format
                            matrix_04[count_04][0], matrix_04[count_04][1] = xywh[0], int(cls)
                            count_04 += 1

                        if save_img or save_crop or view_img:  # Add bbox to image
                            c = int(cls)  # integer class
                            label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                            annotator.box_label(xyxy, label, color=colors(c, True))
                        if save_crop:
                            c = int(cls)  # integer class
                            save_one_box(xyxy, imc, file=save_dir / 'crops' / f'{names[c]}' / f'{p.stem}.jpg', BGR=True)
                    # -----------目标对象判定数据读取结束-----------------------------------#

            # -----------记录1柜指示灯情况记录-----------------------------------#
            if count == 3 and flag1:
                # -----------记录1柜指示灯情况录入txt文件-------------------#
                matrix_01_np = np.array(matrix_01)
                matrix_01_np = matrix_01_np[np.lexsort(matrix_01_np[:, ::-1].T)]
                matrix_01_np = matrix_01_np.tolist()
                for j in range(2):
                    line = int(matrix_01_np[j][1])
                    with open('./res_info/1.txt', 'a') as f:
                        if j == 1:
                            f.write((str(line)+'\n'))
                        else:
                            f.write((str(line)+ ' '))
                # -----------记录1柜指示灯情况录入txt文件结束-----------------#

                if matrix_01_np[0][1] == 2 and matrix_01_np[1][1] == 3:
                    cabinet1_list.append(0)
                else:
                    cabinet1_list.append(1)
            else:
                cabinet1_list.append(0)
            # -----------记录1柜指示灯情况结束-----------------------------------#

            # -----------记录4柜指示灯情况-----------------------------------#
            if count == 6 and flag2:
                # -----------记录4柜指示灯情况录入txt文件-------------------#
                matrix_04_np = np.array(matrix_04)
                matrix_04_np = matrix_04_np[np.lexsort(matrix_04_np[:, ::-1].T)]
                matrix_04_np = matrix_04_np.tolist()
                for j in range(5):
                    line = int(matrix_04_np[j][1])
                    with open('./res_info/4.txt', 'a') as f:
                        if j == 4:
                            f.write((str(line) + '\n'))
                        else:
                            f.write((str(line)+ ' '))
                # -----------记录4柜指示灯情况录入txt文件结束-----------------#

                if matrix_04_np[0][1] == 2 and matrix_04_np[1][1] == 2 and matrix_04_np[2][1] == 2 \
                        and matrix_04_np[3][1] == 3 and matrix_04_np[4][1] == 3:
                    cabinet4_list.append(0)
                else:
                    cabinet4_list.append(1)
            else:
                cabinet4_list.append(0)
            # -----------记录4柜指示灯情况结束-----------------------------------#



            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

    code_path = os.path.dirname(os.path.realpath(__file__))  # 10.18


    # -----------记录1柜异常-----------------------------------#
    if 1 in cabinet1_list:
        print("--------------------------")
        # 获取当前的时间
        import datetime
        current_time = datetime.datetime.now()

        # 获取渗水置信度最大的那个帧号
        cabinet1_light_error_frame_id = cabinet1_list.index(max(cabinet1_list))

        filename = 'cabinet1_light_error.txt'

        cabinet1_light_error_path = str(save_dir / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
        with open(cabinet1_light_error_path + filename, 'a') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
            f.write(str(current_time))  # 写入当前时间
            f.write('\n')
            f.write(code_path)  # 10.18
            f.write('/')  # 10.18
            f.write(str(save_dir))  # 写入路径
            f.write('\n')
            f.write('机柜一')  # 写入当前的site  10.18
            f.write('\n')  # 10.18
            f.write('cabinet1_light_error')  # 写入当前的类别
            f.write('\n')
            f.write(str(cabinet1_light_error_frame_id))  # 写入当前帧号
            f.write('\n')
            f.write('cabinet1_light_error')  # 写入当前图片名字  10.18
            f.write('_')  # 写入当前图片名字
            f.write(str(cabinet1_light_error_frame_id))  # 写入当前图片名字
            f.write('.jpg')  # 写入当前图片名字
            f.write('\n')

        dict = {}
        dict['time'] = str(current_time)
        dict['path'] = code_path + '/' + str(save_dir)  # 10.18
        dict['site'] = '机柜一'  # 10.18
        dict['class'] = 'cabinet1_light_error'
        dict['frame'] = str(cabinet1_light_error_frame_id)
        dict['img_name'] = 'cabinet1_light_error' + '_' + str(cabinet1_light_error_frame_id) + '.jpg'
        dict['desc'] = '机柜一指示灯异常预警'
        print(dict)
        return dict

    # -----------记录1柜异常结束-----------------------------------#

    # -----------记录4柜异常-----------------------------------#
    if 1 in cabinet4_list:
        # 获取当前的时间
        import datetime
        current_time = datetime.datetime.now()

        # 获取渗水置信度最大的那个帧号
        cabinet4_light_error_frame_id = cabinet4_list.index(max(cabinet4_list))

        filename = 'cabinet4_light_error.txt'

        cabinet4_light_error_path = str(save_dir / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
        with open(cabinet4_light_error_path + filename, 'a') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
            f.write(str(current_time))  # 写入当前时间
            f.write('\n')
            f.write(code_path)  # 10.18
            f.write('/')  # 10.18
            f.write(str(save_dir))  # 写入路径
            f.write('\n')
            f.write('机柜四')  # 写入当前的site  10.18
            f.write('\n')  # 10.18
            f.write('cabinet4_light_error')  # 写入当前的类别
            f.write('\n')
            f.write(str(cabinet4_light_error_frame_id))  # 写入当前帧号
            f.write('\n')
            f.write('cabinet4_light_error')  # 写入当前图片名字  10.18
            f.write('_')  # 写入当前图片名字
            f.write(str(cabinet4_light_error_frame_id))  # 写入当前图片名字
            f.write('.jpg')  # 写入当前图片名字
            f.write('\n')

        dict = {}
        dict['time'] = str(current_time)
        dict['path'] = code_path + '/' + str(save_dir)  # 10.18
        dict['site'] = '机柜四'  # 10.18
        dict['class'] = 'cabinet1_light'
        dict['frame'] = str(cabinet4_light_error_frame_id)
        dict['img_name'] = 'cabinet4_light_error' + '_' + str(cabinet4_light_error_frame_id) + '.jpg'
        dict['desc'] = '机柜四指示灯异常预警'
        print(dict)
        return dict
    # -----------记录4柜异常结束-----------------------------------#
    res = {}
    return res



# -----------异常帧抓取-----------------------------------#
def get_frame_from_video(video_name, frame_id, img_dir, img_name):
    """
    get a specific frame of a video by time in milliseconds
    :param video_name: video name
    :param frame_time: time of the desired frame
    :param img_dir: path which use to store output image
    :param img_name: name of output image
    :return: None
    """
    vidcap = cv2.VideoCapture(video_name)
    # Current position of the video file in milliseconds.
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    # read(): Grabs, decodes and returns the next video frame
    success, image = vidcap.read()

    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    if success:
        # save frame as JPEG file
        cv2.imwrite(img_dir+img_name+'.jpg', image)
# -----------异常帧抓取结束-----------------------------------#

def parse_opt(Weight, Source):
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=Weight, help='model path or triton URL')
    parser.add_argument('--source', type=str, default=Source, help='file/dir/URL/glob/screen/0(webcam)')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[640], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.88, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    parser.add_argument('--vid-stride', type=int, default=1, help='video frame-rate stride')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(Weight, Source):
    opt = parse_opt(Weight=Weight, Source=Source)
    check_requirements(exclude=('tensorboard', 'thop'))
    res = run(**vars(opt))
    if res:
        # -----------异常帧保存-----------------------------------#
        code_path = os.path.dirname(os.path.realpath(__file__))  # 10.18
        path = res.get('path')
        class_name = res.get('class')
        frame_id = int(res.get('frame'))

        video_path = path + '/'
        img_name = class_name + '_' + str(frame_id)
        for root, dirs, files in os.walk(video_path):
            for name in files:
                if '.mp4' or '.MP4' in name:
                    video_name = os.path.join(root, name)
                    img_dir = root
                    get_frame_from_video(video_name, frame_id, img_dir, img_name)
    return res


if __name__ == "__main__":
    Root = '/root/yolov5'
    opt = parse_opt()
    res = main(opt)
    if res:
        path = res.get('path')
        class_name = res.get('class')
        frame_id = int(res.get('frame'))

        video_path = './' + path + '/'
        img_name = class_name + '_' + str(frame_id)
        for root, dirs, files in os.walk(video_path):
            for name in files:
                if '.mp4' in name:
                    video_name = os.path.join(root, name)
                    img_dir = root
        get_frame_from_video(video_name, frame_id, img_dir, img_name)



