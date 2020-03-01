import torch as ch
from robustness.datasets import CIFAR
from robustness.model_utils import make_and_restore_model
import sys
import numpy as np

ATTACK_EPS = eval(sys.argv[-1])
ATTACK_STEPS = int(sys.argv[-2])
ATTACK_STEPSIZE = 2.5*ATTACK_EPS/ATTACK_STEPS
CONSTRAINT = sys.argv[-3]
model_path = sys.argv[-4]
label_dim = int(sys.argv[-5])

if label_dim > 0:
	ds = CustomCIFAR(label_dim, '/home/zhuzby/data')
else:
	ds = CIFAR('/opt/harry/data')
model, _ = make_and_restore_model(arch='resnet50', dataset=ds,
             resume_path=model_path)
model.eval()

_, test_loader = ds.make_loaders(workers=8, batch_size=128, shuffle_val=False)

kwargs = {
    'constraint': CONSTRAINT, # use L2-PGD
    'eps': ATTACK_EPS, # L2 radius around original image
    'step_size': ATTACK_STEPSIZE,
    'iterations': ATTACK_STEPS,
    'do_tqdm': True,
    # 'custom_loss': custrom_train_loss,
}

# idx = np.arange(label_dim)
# order = np.load('../data/rnd_label_c10_5.npy')[idx].T
# train_crit = torch.nn.BCELoss()
# def custom_train_loss(logits, targ):
#     if torch.cuda.is_available():
#         targets = torch.from_numpy(order[targ.cpu().numpy()]).cuda()
#     else:
#         targets = torch.from_numpy(order[targ.numpy()])
#     outputs = torch.sigmoid(logits.float())
#     return train_crit(outputs.float(), targets.float())
# train_args.custom_train_loss = custom_train_loss

advs = []
labels = []
for im, label in test_loader:
	_, im_adv = model(im, label, make_adv=True, **kwargs)
	advs = im_adv.cpu().numpy() if len(advs)==0 else np.vstack((advs, im_adv.cpu().numpy()))
	labels = label.cpu().numpy() if len(labels) == 0 else np.hstack((labels, label.cpu().numpy()))
np.save('advs/adv_l{}_{}_{}.npy'.format(CONSTRAINT, ATTACK_STEPS, ATTACK_EPS), advs)
np.save('advs/labels_adv_l{}_{}_{}.npy'.format(CONSTRAINT, ATTACK_STEPS, ATTACK_EPS), advs)


# from robustness.tools.vis_tools import show_image_row
# from robustness.tools.label_maps import CLASS_DICT

# # Get predicted labels for adversarial examples
# pred, _ = model(im_adv)
# label_pred = ch.argmax(pred, dim=1)

# # Visualize test set images, along with corresponding adversarial examples
# show_image_row([im.cpu(), im_adv.cpu()],
#          tlist=[[CLASS_DICT['CIFAR'][int(t)] for t in l] for l in [label, label_pred]],
#          fontsize=18,
#          filename='./adversarial_example_CIFAR.png')
