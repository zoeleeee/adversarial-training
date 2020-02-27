import torch as ch
from robustness.datasets import CIFAR
from robustness.model_utils import make_and_restore_model

ATTACK_EPS = float(sys.argv[-1])
ATTACK_STEPS = int(sys.argv[-1])
ATTACK_STEPSIZE = 2.5*ATTACK_EPS/ATTACK_STEPSIZE


ds = CIFAR('/opt/harry/data')
model, _ = make_and_restore_model(arch='resnet50', dataset=ds,
             resume_path='trained_models')
model.eval()

_, test_loader = ds.make_loaders(workers=8,
                                    batch_size=128)
_, (im, label) = next(enumerate(test_loader))

kwargs = {
    'constraint':'2', # use L2-PGD
    'eps': ATTACK_EPS, # L2 radius around original image
    'step_size': ATTACK_STEPSIZE,
    'iterations': ATTACK_STEPS,
    'do_tqdm': True,
}

_, im_adv = model(im, label, make_adv=True, **kwargs)
np.save('adv_exs.npy', im_adv)


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