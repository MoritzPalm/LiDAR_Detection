import os

import numpy as np
import time
import argparse

import cv2

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, UnNormalizer, Normalizer


print('CUDA available: {}'.format(torch.cuda.is_available()))


def main(args=None):

	validation_path = "data/NAPLab-LiDAR/annotations_val.csv"
	class_list_path = "data/NAPLab-LiDAR/class_names.csv"
	final_model_path = "model_final.pt"
	dataset_val = CSVDataset(train_file=validation_path, class_list=class_list_path, transform=transforms.Compose([Normalizer(), Resizer()]))
	sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=2, drop_last=False)
	dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

	retinanet = torch.load(final_model_path)

	use_gpu = True

	if use_gpu:
		if torch.cuda.is_available():
			retinanet = retinanet.cuda()

	if torch.cuda.is_available():
		retinanet = torch.nn.DataParallel(retinanet).cuda()
	else:
		retinanet = torch.nn.DataParallel(retinanet)

	retinanet.eval()

	unnormalize = UnNormalizer()

	output_dir = "output_images"
	os.makedirs(output_dir, exist_ok=True)

	def draw_caption(image, box, caption):

		b = np.array(box).astype(int)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
		cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

	for idx, data in enumerate(dataloader_val):

		with torch.no_grad():
			st = time.time()
			if torch.cuda.is_available():
				scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
			else:
				scores, classification, transformed_anchors = retinanet(data['img'].float())
			print('Elapsed time: {}'.format(time.time()-st))
			idxs = np.where(scores.cpu()>0.5)
			img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

			img[img<0] = 0
			img[img>255] = 255

			img = np.transpose(img, (1, 2, 0))

			img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

			for j in range(idxs[0].shape[0]):
				bbox = transformed_anchors[idxs[0][j], :]
				x1 = int(bbox[0])
				y1 = int(bbox[1])
				x2 = int(bbox[2])
				y2 = int(bbox[3])
				label_name = dataset_val.labels[int(classification[idxs[0][j]])]
				draw_caption(img, (x1, y1, x2, y2), label_name)

				cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
				print(label_name)

			# cv2.imshow('img', img)
			# cv2.waitKey(0)
			output_path = os.path.join(output_dir, f"image_{idx}.jpg")
			cv2.imwrite(output_path, img)

			print(f"Image saved: {output_path}")



if __name__ == '__main__':
 main()