from robustness.model_utils import make_and_restore_model
from robustness.datasets import CIFAR
import sys
import numpy as np
import os
from scipy.special import softmax, expit
import torch
from torch.utils.data import TensorDataset, DataLoader

def check_normal(preds, labels):
	pred_labels = np.argmax(softmax(preds, axis=-1), axis=-1)
	print('acc:', np.mean(pred_labels==labels))

def check_hamming(scores, labels, t, name):
	scores = expit(scores)
	if not os.path.exists('eval/hamming/hamming_labels_{}.npy'.format(name)):
		idxs = np.arange(10)
		rep = np.load('../data/rnd_label_c10_5.npy')
		samples = np.array([[1 if u >= 0.5 else 0 for u in v] for v in scores])
		preds, preds_dist, preds_score = [], [], []
		for i in np.arange(samples.shape[0]):
			_pred = np.repeat(samples[i].reshape((-1, len(samples[i]))), rep.shape[0], axis=0)
			dists = np.sum(np.absolute(_pred - rep), axis=1)
			min_dist = np.min(dists)
			pred_labels = np.arange(len(dists))[dists==min_dist]
			pred_scores = [np.sum([scores[i][k] if samples[i][k] == rep[j][k] else 1-scores[i][k] for k in np.arange(len(scores[i]))]) for j in pred_labels]
			pred_label = pred_labels[np.argmax(pred_scores)]
			preds.append(pred_label)
			preds_dist.append(dists[pred_label])
			preds_score.append(np.max(pred_scores))	
		np.save('eval/hamming/hamming_labels_{}.npy'.format(name), preds)
		np.save('eval/hamming/hamming_labels_dists_{}.npy'.format(name), preds_dist)
		# np.save('eval/hamming/hamming_labels_score_{}.npy'.format(name), preds_score)
		preds = np.array(preds)
		preds_dist = np.array(preds_dist)
	else:
		preds = np.load('eval/hamming/hamming_labels_{}.npy'.format(name))
		preds_dist = np.load('eval/hamming/hamming_labels_dists_{}.npy'.format(name))
	print('avg Hamming distance:{}, max:{}, min:{}, med:{}'.format(np.mean(preds_dist), np.max(preds_dist), np.min(preds_dist), np.median(preds_dist)))
	print(t, 'acc:', np.sum(preds_dist[preds == labels] < t) / len(labels))
	print(t, 'acc:', np.sum(preds_dist[preds != labels] < t) / len(labels))



def main():
	path = sys.argv[-1]
	metrics = sys.argv[-2]

	name = 'pred'
	if sys.argv[-3].startswith('advs'):
		name+= '_advs'

	if not os.path.exists('eval/{}_{}.npy'.format(name, path.split('/')[-1][:-3])):
		ds = CIFAR('/opt/harry/data')
		model, _ = make_and_restore_model(arch='resnet50', dataset=ds, resume_path=path)
		model = model.eval()
		if sys.argv[-3].startswith('advs'):
			im_adv = np.load(sys.argv[-3])
			labs = np.load(os.path.join(sys.argv[-3].split('/')[0], 'labels_'+sys.argv[-3].split('/')[1]))
			data = TensorDataset(torch.tensor(im_adv), torch.tensor(labs))
			test_loader = DataLoader(data, batch_size=128, num_workers=8, shuffle=False)
		else:
			_, test_loader = ds.make_loaders(workers=8, batch_size=128)
		preds, labels = [], []
		for i, (im, label) in enumerate(test_loader):
			output, _ = model(im)
			label = label.cpu().numpy()
			preds = output.detach().cpu().numpy() if len(preds)==0 else np.vstack((preds, output.detach().cpu().numpy()))
			labels = label if len(labels)==0 else np.hstack((labels, label))
		np.save('eval/{}_{}.npy'.format(name, path.split('/')[-1][:-3]), preds)
		np.save('eval/label_{}.npy'.format(path.split('/')[-1][:-3]), labels)
	else:
		preds = np.load('eval/{}_{}.npy'.format(name, path.split('/')[-1][:-3]))
		labels = np.load('eval/label_{}.npy'.format(path.split('/')[-1][:-3]), labels)

	if metrics == 'origin':
		check_normal(preds, labels)
	elif metrics == 'hamming':
		check_hamming(preds, labels, int(sys.argv[-4]), path.split('/')[-1][:-3])



if __name__ == '__main__':
	main()
