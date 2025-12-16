from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

log_dir = './tensorboard_logs_batch8/version_0'
ea = EventAccumulator(log_dir)
ea.Reload()

val_acc = ea.Scalars('valid/acc')
train_acc = ea.Scalars('train/acc_epoch')
val_loss = ea.Scalars('valid/loss')

epochs_val = [(int(s.step), s.value) for s in val_acc]
epochs_train = [(int(s.step), s.value) for s in train_acc]
epochs_loss = [(int(s.step), s.value) for s in val_loss]

print('Epoch | Train Acc | Val Acc   | Val Loss')
print('-' * 45)
min_len = min(len(epochs_val), len(epochs_train), len(epochs_loss))
for e in range(min_len):
    print(f'{epochs_val[e][0]:5d} | {epochs_train[e][1]:9.4f} | {epochs_val[e][1]:.4f} | {epochs_loss[e][1]:.4f}')

best_idx = max(range(len(epochs_val)), key=lambda i: epochs_val[i][1])
print(f'\nBest: {epochs_val[best_idx][1]:.4f} at epoch {epochs_val[best_idx][0]}')
print(f'Final: {epochs_val[-1][1]:.4f} at epoch {epochs_val[-1][0]}')

if len(epochs_val) >= 10:
    last_10 = [x[1] for x in epochs_val[-10:]]
    oscillation = max(last_10) - min(last_10)
    print(f'\nOscillation (last 10 epochs): {oscillation:.4f}')

# Analyze train-val gap
train_val_gap = [(epochs_train[i][1] - epochs_val[i][1]) for i in range(min_len)]
avg_gap = sum(train_val_gap) / len(train_val_gap)
print(f'Average train-val gap: {avg_gap:.4f}')
