from model.base_model import BaseModel
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import softmax
import torch 
from typing import List
from scipy.stats import ttest_1samp

class ContrastiveEvaluator:
    def __init__(self, model: BaseModel, dataset: Dataset, contrast_texts: List[str], batch_size = 32, pvalue=0.05):
        self.model = model 
        self.dataset = dataset
        self.batch_size = batch_size
        self.contrast_texts = contrast_texts
        self.pvalue = pvalue

    def evaluate(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        correct, tp, fp, tn, fn = 0, 0, 0, 0, 0

        with torch.no_grad():
            contrast_features = self.model.get_text_features(self.contrast_texts)

            for batch in dataloader:
                image_features = self.model.get_image_features(batch[0].unbind(0))
                text_features = self.model.get_text_features(batch[1][0])
                target_similarity = torch.diagonal(image_features @ text_features.t())
                contrast_similarities = image_features @ contrast_features.t()

                num_items = batch[0].shape[0]
                num_contrast_texts = len(self.contrast_texts)

                scores_combined = torch.stack([
                    target_similarity.repeat(num_contrast_texts), 
                    contrast_similarities.flatten()
                ])
                probs = softmax(scores_combined, dim=0)[0,].reshape(num_contrast_texts,num_items).t().cpu().detach()
                pvalues = ttest_1samp(probs, 0.5, alternative='greater', axis=1).pvalue
                labels = batch[2].numpy()
                
                correct += sum((pvalues <= self.pvalue) == labels)
                tp += sum((pvalues <= self.pvalue) & labels)
                fp += sum((pvalues <= self.pvalue) & ~labels)
                tn += sum((pvalues > self.pvalue) & ~labels)
                fn += sum((pvalues > self.pvalue) & labels)
                
                del image_features, text_features, scores_combined, probs, pvalues

        print(f"Accuracy: {correct/self.dataset.__len__()*100:.3f}%")
        print(f"Precision: {tp/(tp+fp):.3f}")
        print(f"Recall: {tp/(tp+fn):.3f}")

        del contrast_features
        torch.cuda.empty_cache()

        return {
            "accuracy": correct/self.dataset.__len__()*100,
            "precision": tp/(tp+fp),
            "recall": tp/(tp+fn)
        }
    