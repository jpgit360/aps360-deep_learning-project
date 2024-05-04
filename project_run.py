def get_test_result(net, bs, lr, ep, img, signs_list, path):
    model_path = path + get_model_name(net.name, batch_size=bs, learning_rate=lr, epoch=ep)
    state = torch.load(model_path)
    net.load_state_dict(state)

    softmax = nn.Softmax(dim=1)
    output = softmax(net(img))
    max_value, predicted_output = torch.max(output, 1)
    sign_index = predicted_output.item()
    return (output, float(max_value), signs_list[sign_index])

def img_to_tensor(img, transform):
        tensor_img = transform(img).unsqueeze(0)
        return tensor_img

def run_project(signs_list, transform, input_path, yolo_model_path, cnn_model_path, net, bs, lr, ep):
    from ultralytics import YOLO

    yolo_model = YOLO(yolo_model_path)
    master_list = []

    for filename in os.listdir(input_path):
        if filename.endswith('.jpg') or filename.endswith('.JPEG') or filename.endswith('.png'):
            input_img = PIL.Image.open(input_path + filename).convert("RGB")
            # reduce input_img size
            resized_input_img = input_img.resize((input_img.width // 4, input_img.height // 4))
            result = yolo_model(input_img)
            package_list = []
            for box in result[0].boxes:
                if result[0].names[box.cls.item()]:
                    xyxy = box.xyxy.tolist()
                    cropped_img = input_img.crop(xyxy[0])
                    cropped_img = cropped_img.resize((150,150))
                    yolo_output_img = img_to_tensor(cropped_img, transform)
                    output, percentage, sign_type = get_test_result(net=net,
                                                                    bs=bs,
                                                                    lr=lr,
                                                                    ep=ep,
                                                                    img=yolo_output_img,
                                                                    signs_list=signs_list,
                                                                    path=cnn_model_path
                                                                    )
                    package_list.append((cropped_img, output, percentage, sign_type))
                    break
            master_list.append((resized_input_img, package_list))
    '''
    output is of this form
    [
        resized_input_img_1, [ (cropped_img_1, output_1, percentage_1, sign_type_1) ],
        resized_input_img_1, [ (cropped_img_2, output_2, percentage_2, sign_type_2) ],
        ...
        ...
    ]
    '''
    return master_list

def print_project(master_list):
    for input_imgs in range(len(master_list)):
        print("---------------------")
        print("Original Input Image:")
        print("---------------------")
        display(master_list[input_imgs][0]) # original image
        for yolo_cropped in range(len(master_list[input_imgs][1])):
            print("---------------------")
            print("Output:")
            print("---------------------")
            display(master_list[input_imgs][1][yolo_cropped][0]) # yolo cropped image
            #print(master_list[input_imgs][1][yolo_cropped][1]) # full output (for debugging purposes)
            percentage = 100 * master_list[input_imgs][1][yolo_cropped][2] # percentage it is sure
            sign_type = master_list[input_imgs][1][yolo_cropped][3] # sign type prediction
            print(f"The model is {percentage}% sure it is {sign_type}.")


signs_dict = {
        'chevron': 0,
        'parking': 1,
        'pedestrian_cross': 2,
        'bicycle_only': 3,
        'keep_right': 4,
        'speed_limit': 5,
        'no_entry': 6,
        'no_heavy_vehicles': 7,
        'no_left_turn': 8,
        'no_overtaking': 9,
        'no_parking': 10,
        'no_right_turn': 11,
        'no_uturn': 12,
        'one_way_left': 13,
        'one_way_right': 14,
        'one_way_straight': 15,
        'stop': 16,
        'yield': 17,
        'children': 18,
        'curve_left': 19,
        'curve_right': 20,
        'road_bump': 21,
        'school_zone': 22,
        'stop_ahead': 23,
        'traffic_signal_ahead': 24,
        'merge_right': 25
}
signs_list = list(signs_dict.keys())

transform = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

input_path = "/content/gdrive/MyDrive/Project/test/" # path where all input images are
yolo_model_path = "/content/gdrive/MyDrive/Project/best.pt" #path where yolo model is
cnn_model_path = "/content/gdrive/MyDrive/Project/training_output2/model_cnnv6_bs32_lr0.001_2024-04-06_00:49:07/" #path where cnn model is

#specify net, bs, lr, ep, depending on the model name
master_list = run_project(signs_list, transform, input_path, yolo_model_path, cnn_model_path, net=CNN_v6(), bs=32, lr=0.001, ep=39)
# print out our inputs and corresponding outputs
print_project(master_list)