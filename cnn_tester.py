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

def confusion_mat(predicted_list, true_list, pred, true):
    count = 0
    for i in range(0,len(predicted_list)):
        if predicted_list[i] == pred and true_list[i] == true:
            count += 1
    return count

def get_test_accuracy(net, bs, lr, ep, path):
    model_path = path + get_model_name(net.name, batch_size=bs, learning_rate=lr, epoch=ep)
    state = torch.load(model_path)
    net.load_state_dict(state)
    criterion = nn.CrossEntropyLoss()
    test_err, test_loss, pred_list, true_list = evaluate_test(net, test_loader, criterion)

    return pred_list, true_list, 1 - test_err

def print_confusion_mat(pred_list, true_list, num_classes):
    for i in signs_list:
        print(i, end=', ')
    print("")
    for i in range(0, num_classes):
        print(signs_list[i], end=', ')
        for j in range(0, num_classes):
            print(confusion_mat(pred_list, true_list, pred=i, true=j), end=', ')
        print("")

def confusion_mat_to_csv(pred_list, true_list, num_classes):
    import copy
    import csv
    first_row = copy.deepcopy(signs_list)
    first_row.insert(0, "")
    mat = []
    mat.append(first_row)

    for i in range(0, num_classes):
        row = []
        row.append(signs_list[i])
        for j in range(0, num_classes):
            row.append(confusion_mat(pred_list, true_list, pred=i, true=j))
        mat.append(row)

    np_mat = np.array(mat)
    csv_file_path = "output.csv"
    with open(csv_file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(np_mat)

path = "/content/gdrive/MyDrive/APS360-Notebooks/Project/training_output2/model_cnnv6_bs32_lr0.001_2024-04-06_00:49:07/"
pred_list, true_list, test_acc = get_test_accuracy(net=CNN_v6(), bs=32, lr=0.001, ep=39, path=path)
print("The test accuracy is: ", test_acc)

confusion_mat_to_csv(pred_list, true_list, 26)