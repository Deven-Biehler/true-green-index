import matplotlib.pyplot as plt

# Epochs
epochs = list(range(1, 26))

# Training and validation loss
train_loss = [0.3476, 0.2717, 0.2091, 0.2026, 0.1829, 0.1791, 0.1778, 0.1297, 0.1203, 0.1155, 
              0.1110, 0.1134, 0.1074, 0.0991, 0.0990, 0.0968, 0.0996, 0.0967, 0.1025, 0.1024, 
              0.0969, 0.0886, 0.0994, 0.0908, 0.0999]
val_loss = [0.1133, 0.2115, 0.1088, 0.1236, 0.3271, 0.2549, 0.1798, 0.1081, 0.1122, 0.1224, 
            0.0950, 0.1154, 0.0941, 0.1150, 0.1151, 0.1179, 0.1108, 0.1187, 0.1149, 0.1167, 
            0.1314, 0.1127, 0.1125, 0.1101, 0.1083]

# Training and validation accuracy
train_acc = [0.8674, 0.9043, 0.9225, 0.9320, 0.9407, 0.9442, 0.9449, 0.9587, 0.9599, 0.9634, 
             0.9647, 0.9652, 0.9667, 0.9680, 0.9673, 0.9689, 0.9682, 0.9684, 0.9678, 0.9653, 
             0.9705, 0.9673, 0.9682, 0.9662, 0.9687]
val_acc = [0.9451, 0.9268, 0.9573, 0.9512, 0.8476, 0.8720, 0.9329, 0.9634, 0.9634, 0.9573, 
           0.9573, 0.9512, 0.9573, 0.9756, 0.9573, 0.9573, 0.9756, 0.9512, 0.9573, 0.9573, 
           0.9390, 0.9573, 0.9573, 0.9634, 0.9634]

# Plotting the loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_loss, label='Train Loss', marker='o')
plt.plot(epochs, val_loss, label='Val Loss', marker='o')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Plotting the accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, label='Train Acc', marker='o')
plt.plot(epochs, val_acc, label='Val Acc', marker='o')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()


# Save the plot
plt.savefig('results/training_results.png')

plt.tight_layout()
plt.show()
