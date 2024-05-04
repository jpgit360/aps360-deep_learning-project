def train_net(net, train_loader, val_loader, batch_size=64, learning_rate=0.01, num_epochs=30, path=""):
    from datetime import datetime
    import pytz
    desired_timezone = 'US/Eastern'
    current_timestamp = datetime.now(pytz.timezone(desired_timezone))
    timestamp_string = current_timestamp.strftime("%Y-%m-%d_%H:%M:%S")

    torch.manual_seed(1000)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)

    train_err = np.zeros(num_epochs)
    train_loss = np.zeros(num_epochs)
    val_err = np.zeros(num_epochs)
    val_loss = np.zeros(num_epochs)

    start_time = time.time()
    for epoch in range(num_epochs):
        total_train_loss = 0.0
        total_train_err = 0.0
        total_epoch = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs)
            notonehot_labels = torch.argmax(labels, 1)

            loss = criterion(outputs, notonehot_labels)
            loss.backward()
            optimizer.step()

            max_value, predicted_output = torch.max(outputs, 1)
            true_output = torch.argmax(labels, 1)
            corr = (predicted_output != true_output)
            total_train_err += int(corr.sum())
            total_train_loss += loss.item()
            total_epoch += len(labels)

        train_err[epoch] = float(total_train_err) / total_epoch
        train_loss[epoch] = float(total_train_loss) / (i+1)
        val_err[epoch], val_loss[epoch] = evaluate(net, val_loader, criterion)
        print(("Epoch {}: Train err: {}, Train loss: {} |"+
               "Validation err: {}, Validation loss: {}").format(
                   epoch + 1,
                   train_err[epoch],
                   train_loss[epoch],
                   val_err[epoch],
                   val_loss[epoch]))

        directory = path + get_dir(net.name, batch_size, learning_rate, timestamp_string)
        if not os.path.exists(directory):
            os.makedirs(directory)

        model_path = directory + get_model_name(net.name, batch_size, learning_rate, epoch)
        torch.save(net.state_dict(), model_path)
    print('Finished Training')
    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Total time elapsed: {:.2f} seconds".format(elapsed_time))

    epochs = np.arange(1, num_epochs + 1)
    np.savetxt("{}_train_err.csv".format(model_path), train_err)
    np.savetxt("{}_train_loss.csv".format(model_path), train_loss)
    np.savetxt("{}_val_err.csv".format(model_path), val_err)
    np.savetxt("{}_val_loss.csv".format(model_path), val_loss)