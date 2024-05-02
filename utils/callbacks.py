import pytorch_lightning as pl
import torch
import torchvision.utils
import os
import numpy as np
import torch.nn.functional as F
# from losses import ArcFace
from pyeer.eer_info import get_eer_stats
from utils.ploters.pyeer.eer_info import get_eer_stats
import random
from numpy.linalg import norm
import matplotlib.pyplot as plt
import json

class CallBackVerification(pl.Callback):
    def __init__(self, data_module):
        super().__init__()
        self.data_module = data_module

    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            dataset = {}
            for batch in self.data_module.val_dataloader():
                x, l = batch
                x = x.to(device=pl_module.device)
                l = l.to(device=pl_module.device)
                features = pl_module(x)
                features = F.normalize(features)
                features = features.detach().cpu().numpy()
                label = l.detach().cpu().numpy()
                for i in range(label.shape[0]):
                    if label[i] in dataset:
                        dataset[label[i]].append(features[i])
                    else:
                        dataset[label[i]] = [features[i]]
            
            gen, imp = self.perform_verification(dataset)
            stat = get_eer_stats(gen, imp)
            eer = torch.tensor(stat.eer*100, dtype=torch.float)
            fmr10 = torch.tensor(stat.fmr10*100, dtype=torch.float)
            fmr20 = torch.tensor(stat.fmr20*100, dtype=torch.float)
            fmr100 = torch.tensor(stat.fmr100*100, dtype=torch.float)

            del dataset
            pl_module.log_dict({'EER_VER': eer, 'FMR20': fmr20, 'FMR10': fmr10, 'FMR100': fmr100})

    
    def on_test_epoch_end(self, trainer, pl_module):
        dataset = {}
        for batch in self.data_module:
            x, l = batch
            x = x.to(device=pl_module.device)
            l = l.to(device=pl_module.device)
            features = pl_module(x)
            features = F.normalize(features)
            features = features.detach().cpu().numpy()
            label = l.detach().cpu().numpy()
            for i in range(label.shape[0]):
                if label[i] in dataset:
                    dataset[label[i]].append(features[i])
                else:
                    dataset[label[i]] = [features[i]]
        
        gen, imp = self.perform_verification(dataset)
        stat = get_eer_stats(gen, imp)
        eer = torch.tensor(stat.eer*100, dtype=torch.float)
        fmr10 = torch.tensor(stat.fmr10*100, dtype=torch.float)
        fmr20 = torch.tensor(stat.fmr20*100, dtype=torch.float)
        fmr100 = torch.tensor(stat.fmr100*100, dtype=torch.float)

        del dataset
        pl_module.log_dict({'EER_VER': eer, 'FMR20': fmr20, 'FMR10': fmr10, 'FMR100': fmr100})


    #verification performance
    @staticmethod
    def perform_verification(dataset):

        keys = list(dataset.keys())
        random.shuffle(keys)
        genuine_list, impostors_list = [], []
        #Compute mated comparisons
        for k in keys:
            for i in range(len(dataset[k]) - 1):
                reference = dataset[k][i]
                for j in range(i + 1, len(dataset[k])):
                    probe = dataset[k][j]
                    value = np.dot(probe,reference)/(norm(probe)*norm(reference))
                    genuine_list.append(value)

        #Compute non-mated comparisons
        for i in range(len(keys)):
            reference = random.choice(dataset[keys[i]])
            for j in range(len(keys)):
                if i != j:
                    probe = random.choice(dataset[keys[j]])
                    value = np.dot(probe,reference)/(norm(probe)*norm(reference))
                    impostors_list.append(value)

        return genuine_list, impostors_list
    

class CallBackSimilarityExtraction(pl.Callback):
    def __init__(self, data_module, output_folder):
        super().__init__()
        self.data_module = data_module
        self.output_folder = output_folder
    
    
    def on_test_epoch_end(self, trainer, pl_module):
        dataset = {}
        for idx, batch in enumerate(self.data_module):
            x, l = batch
            x = x.to(device=pl_module.device)
            l = l.to(device=pl_module.device)
            features = pl_module(x)
            features = F.normalize(features)
            features = features.detach().cpu().numpy()
            # idx = idx.detach().cpu().numpy()
            label = l.detach().cpu().numpy()
            for i in range(label.shape[0]):
                # filename = self.data_module.dataset.images[idx[i]]
                filename = self.data_module.dataset.imgs[idx*64 + i][0]
                if label[i] in dataset:
                    dataset[label[i]].append((filename, features[i]))
                else:
                    dataset[label[i]] = [(filename, features[i])]
        
        gen, imp = self.perform_verification(dataset)

        with open(os.path.join(self.output_folder, 'mated_comparisons.json'), 'w', encoding='utf-8') as f:
            json.dump(gen, f, ensure_ascii=False, indent=4)

        with open(os.path.join(self.output_folder, 'non_mated_comparisons.json'), 'w', encoding='utf-8') as f:
            json.dump(imp, f, ensure_ascii=False, indent=4)
    
        del dataset

        stat = get_eer_stats([float(e) for e in gen.values()], [float(e) for e in imp.values()])
        eer = torch.tensor(stat.eer * 100, dtype=torch.float)
        fmr10 = torch.tensor(stat.fmr10 * 100, dtype=torch.float)
        fmr20 = torch.tensor(stat.fmr20 * 100, dtype=torch.float)
        fmr100 = torch.tensor(stat.fmr100 * 100, dtype=torch.float)
        pl_module.log_dict({'EER': eer, 'FMR10': fmr10, 'FMR20': fmr20, 'FMR100': fmr100})


    #verification performance
    @staticmethod
    def perform_verification(dataset):

        keys = list(dataset.keys())
        random.shuffle(keys)
        genuine_list, impostors_list = {}, {}
        #Compute mated comparisons
        for k in keys:
            for i in range(len(dataset[k]) - 1):
                filename_r, reference = dataset[k][i]
                for j in range(i + 1, len(dataset[k])):
                    filename_p, probe = dataset[k][j]
                    value = np.dot(probe,reference)/(norm(probe)*norm(reference))
                    genuine_list['{}#{}'.format(filename_r, filename_p)] = str(value)

        #Compute non-mated comparisons
        for i in range(len(keys)):
            filename_r, reference = random.choice(dataset[keys[i]])
            for j in range(len(keys)):
                if i != j:
                    filename_p, probe = random.choice(dataset[keys[j]])
                    value = np.dot(probe,reference)/(norm(probe)*norm(reference))
                    impostors_list['{}#{}'.format(filename_r, filename_p)] = str(value)

        return genuine_list, impostors_list
    

class CallBackOpenSetIdentification(pl.Callback):
    def __init__(self, data_module, scores_dir: str, pipeline_type: str = 'combined'):
        super().__init__()
        self.data_module = data_module
        self.pipeline_type = pipeline_type
        self.scores_dir = scores_dir

    def on_test_epoch_end(self, trainer, pl_module):
        dataset = {}
        for batch in self.data_module:
            x, l = batch
            x = x.to(device=pl_module.device)
            l = l.to(device=pl_module.device)
            if self.pipeline_type == 'bot' or self.pipeline_type == 'combined':
                _, features = pl_module(x)
            elif self.pipeline_type == 'top':
                features = pl_module(x)
            else:
                raise ValueError('pipeline_type should be one of the following: bot, top, combined')
            features = F.normalize(features)
            features = features.detach().cpu().numpy()
            label = l.detach().cpu().numpy()
            for i in range(label.shape[0]):
                if label[i] in dataset:
                    dataset[label[i]].append(features[i])
                else:
                    dataset[label[i]] = [features[i]]

        gen, imp = self.perform_identification(dataset)

        stat = get_eer_stats(gen, imp)
        eer = torch.tensor(stat.eer * 100, dtype=torch.float)
        fmr10 = torch.tensor(stat.fmr10 * 100, dtype=torch.float)
        fmr20 = torch.tensor(stat.fmr20 * 100, dtype=torch.float)
        fmr100 = torch.tensor(stat.fmr100 * 100, dtype=torch.float)
        pl_module.log_dict({'EER': eer, 'FNIR10': fmr10, 'FNIR20': fmr20, 'FNIR100': fmr100})

        # save the results to a file
        score_path = os.path.join(self.scores_dir, 'open_set_scores.npz')
        np.savez(score_path, gen=gen, imp=imp)

    #verification performance
    @staticmethod
    def perform_identification(dataset, k_fold=10):
        keys = list(dataset.keys())
        random.shuffle(keys)
        k = int(len(keys)/k_fold)
        total_index = list(np.arange(len(keys)))
        genuine_list, impostors_list = [], []

        for i in range(k_fold):
            start = k*i
            end = k*(i + 1)
            impostors_id = {k:dataset[k] for k in keys[start:end]}
            var_iter = set(np.arange(start, end))
            gen_idx = list(set(total_index) - var_iter)
            gen_keys = [keys[k] for k in gen_idx]
            genuines_id = {k:dataset[k] for k in gen_keys}
            enrolled_database, search_gen_database = [], []

            for g in genuines_id:
                aux = random.sample(genuines_id[g], 2)
                enrolled_database.append(aux[0])
                search_gen_database.append(aux[1])

            #Compute mated comparisons
            for s in search_gen_database:
                probe = s
                temp_list = []
                for r in enrolled_database:
                    reference = r
                    value = np.dot(probe,reference)/(norm(probe)*norm(reference))
                    temp_list.append(value)
                temp_list.sort(reverse=True)
                genuine_list.append(temp_list[0])

            #Compute non-mated comparisons
            for nm in impostors_id:
                for v in impostors_id[nm]:
                    probe = v
                    temp_list = []
                    for r in enrolled_database:
                        reference = r
                        value = np.dot(probe,reference)/(norm(probe)*norm(reference))
                        temp_list.append(value)
                    temp_list.sort(reverse=True)
                    impostors_list.append(temp_list[0])

        return genuine_list, impostors_list


class CallbackCloseSetIdentification(pl.Callback):
    def __init__(self, data_module, scores_dir: str, pipeline_type: str = 'combined'):
        super().__init__()
        self.data_module = data_module
        self.pipeline_type = pipeline_type
        self.scores_dir = scores_dir


    def on_test_epoch_end(self, trainer, pl_module):
        dataset = {}
        for batch in self.data_module:
            x, l = batch
            x = x.to(device=pl_module.device)
            l = l.to(device=pl_module.device)
            if self.pipeline_type == 'bot' or self.pipeline_type == 'combined':
                _, features = pl_module(x)
            elif self.pipeline_type == 'top':
                features = pl_module(x)
            else:
                raise ValueError('pipeline_type should be one of the following: bot, top, combined')
            features = F.normalize(features)
            features = features.detach().cpu().numpy()
            label = l.detach().cpu().numpy()
            for i in range(label.shape[0]):
                if label[i] in dataset:
                    dataset[label[i]].append(features[i])
                else:
                    dataset[label[i]] = [features[i]]

        final_scores, final_mated_comparisons = self.perform_identification(dataset)

        # save final scores to txt file
        for fold in range(len(final_scores)):
            score_path = os.path.join(self.scores_dir, 'close_set_scores_{}.txt'.format(fold))
            score_tp_path = os.path.join(self.scores_dir, 'close_set_scores_tp_{}.txt'.format(fold))
            np.savetxt(score_path, final_scores[fold])
            np.savetxt(score_tp_path, final_mated_comparisons[fold])
            print('[INFO] Saved closed set scores to {}'.format(score_path))
            print('[INFO] Saved closed set scores true positive to {}'.format(score_tp_path))

        # stat = get_eer_stats(gen, imp)
        # eer = torch.tensor(stat.eer * 100, dtype=torch.float)
        # fmr10 = torch.tensor(stat.fmr10 * 100, dtype=torch.float)
        # fmr20 = torch.tensor(stat.fmr20 * 100, dtype=torch.float)
        # fmr100 = torch.tensor(stat.fmr100 * 100, dtype=torch.float)
        # pl_module.log_dict({'EER': eer, 'FNIR10': fmr10, 'FNIR20': fmr20, 'FNIR100': fmr100})

    def perform_identification(self, dataset, k_fold=10):
        keys = list(dataset.keys())
        final_scores, final_mated_comparisons = [], []

        for i in range(k_fold):
            random.shuffle(keys)
            mated_comparisons, scores = [], []

            enrolled_database, search_gen_database = [], []
            # creating enrollment and query databases
            for k in keys:
                aux = random.sample(dataset[k], 2)
                enrolled_database.append((k, aux[0]))
                search_gen_database.append((k, aux[1]))

            # Compute mated comparisons
            for k1, s in search_gen_database:
                probe = s

                for k2, r in enrolled_database:
                    reference = r
                    value = np.dot(probe, reference) / (norm(probe) * norm(reference))
                    scores.append((k1, k2, value))

                    if k1 == k2:
                        mated_comparisons.append((k1, k2))

            final_scores.append(scores.copy())
            final_mated_comparisons.append(mated_comparisons.copy())

            # if i > 0:
            #     continue
            #
            # scores_to_plot = scores.copy()
            # scores_to_plot = [x for x in scores_to_plot if x[0] != x[1]]
            # scores_to_plot.sort(key=lambda x: x[2], reverse=True)
            # # get only the entries that has x[0] = final_scores[0][0]
            # refs = [x[0] for x in scores_to_plot[:20]]
            # for rank in range(0, len(refs)):
            #     scores_copy = scores_to_plot.copy()
            #     scores_copy = [x for x in scores_copy if x[0] == refs[rank]]
            #     top_scores = scores_copy[:5]
            #     top_images = [x[1] for x in top_scores]
            #     top_images_labels = [str(x[2]) for x in top_scores]
            #
            #     ref_img = Image.open(self.data_module.dataset.samples[refs[rank]][0])
            #     top_images_imgs = [Image.open(self.data_module.dataset.samples[x][0]) for x in top_images]
            #
            #     images = [ref_img] + top_images_imgs
            #     # resize images to 256x256
            #     images = [img.resize((256, 256)) for img in images]
            #     labels = ['reference'] + top_images_labels
            #     # given images and labels create a plot - images in the first row and labels in the second row
            #     fig, axs = plt.subplots(2, len(images), figsize=(15, 15))
            #     for j in range(len(images)):
            #         axs[0, j].imshow(images[j])
            #         axs[0, j].set_title(labels[j])
            #         axs[0, j].axis('off')
            #     # save figure with name k1_filename
            #     figname = os.path.join(self.run_dir, '{}_closest_matches_{}.jpg'.format(rank,refs[rank]))
            #     plt.savefig(figname, bbox_inches='tight')
            #     plt.close()
            #     rank += 1

        return final_scores, final_mated_comparisons