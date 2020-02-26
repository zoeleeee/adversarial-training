from robustness.model_utils import make_and_restore_model
from robustness.datasets import CIFAR
import sys
import numpy as np
import os
from scipy.special import softmax, expit

def check_normal(preds, labels):
	pred_labels = np.argmax(softmax(preds, axis=-1), axis=-1)
	print('acc:', np.mean(pred_labels==labels))

def check_hamming(scores, labels, t, name):
	scores = expit(scores)
	if not os.path.exists('eval/hamming/hamming_labels_{}.npy'.format(name)):
		idxs = np.arange(10)
		rep = np.load('rnd_label_c10_5.npy')[idxs].T
		samples = np.array([[1 if u >= 0.5 else 0 for u in v] for v in preds])
		preds, preds_dist, preds_score = [], [], []
		for i in np.arange(pred_labels.shape[0]):
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
		preds = np.load('_models/hamming_labels_{}.npy'.format(name))
		preds_dist = np.load('_models/hamming_labels_dists_{}.npy'.format(name))
	print('avg Hamming distance:{}, max:{}, min:{}, med:{}'.format(np.mean(preds_dist), np.max(preds_dist), np.min(preds_dist), np.median(preds_dist)))
	print(t, 'acc:', np.sum(preds_dist[preds == labels] < t) / len(labels))
	print(t, 'acc:', np.sum(preds_dist[preds != labels] < t) / len(labels))



def main():
	path = sys.argv[-1]
	metrics = sys.argv[-2]

	if not os.path.exists('eval/pred_{}.npy'.format(path.split('/')[-1][:-4])):
		ds = CIFAR('/home/zhuzby/data')
		model, _ = make_and_restore_model(arch='resnet50', dataset=ds, resume_path=path)
		model = model.eval()
		_, test_loader = ds.make_loaders(workers=8, batch_size=128)
		preds, labels = [], []
		for i, (im, label) in enumerate(test_loader):
			output, _ = model(im)
			label = label.cpu().numpy()
			preds = output.detach().cpu().numpy() if len(preds)==0 else np.vstack((preds, output.detach().cpu().numpy()))
			labels = label if len(labels)==0 else np.hstack((labels, label))
		np.save('eval/pred_{}.npy'.format(path.split('/')[-1][:-4]), preds)
		np.save('eval/label_{}.npy'.format(path.split('/')[-1][:-4]), labels)
	else:
		preds = np.load('eval/pred_{}.npy'.format(path.split('/')[-1][:-4]))
		labels = np.load('eval/label_{}.npy'.format(path.split('/')[-1][:-4]))

	if metrics == 'origin':
		check_normal(preds, labels)
	elif metrics == 'hamming':
		check_hamming(preds, labels, int(sys.argv[-3]), path.split('/')[-1][:-4])



if __name__ == '__main__':
	main()