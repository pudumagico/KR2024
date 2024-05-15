from model.base_model import BaseModel
from torch.utils.data import Dataset, DataLoader
import torch 

class Evaluator:
    def __init__(self, model: BaseModel, dataset: Dataset, batch_size = 32):
        self.model = model 
        self.dataset = dataset
        self.batch_size = batch_size

    def evaluate(self):
        dataloader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)
        correct, tp, fp, tn, fn = 0, 0, 0, 0, 0

        with torch.no_grad():
            for batch in dataloader:
                logits_per_image = self.model.score(batch[0].unbind(0), [*batch[1][0], *batch[1][1]])
            
                num_items = batch[0].shape[0]
                scores = torch.stack([
                    torch.diagonal(logits_per_image[:,:num_items]), 
                    torch.diagonal(logits_per_image[:,num_items:])
                ])
                probs = torch.nn.functional.softmax(scores, dim=0)
                labels = batch[2].to(self.model.gpu)
                
                correct += sum((probs[0, :] >= probs[1, :]) == labels)
                tp += sum((probs[0, :] >= probs[1, :]) & labels)
                fp += sum((probs[0, :] >= probs[1, :]) & ~labels)
                tn += sum((probs[0, :] < probs[1, :]) & ~labels)
                fn += sum((probs[0, :] < probs[1, :]) & labels)
                
                del logits_per_image, scores, probs

        print(f"Accuracy: {correct/self.dataset.__len__()*100:.3f}%")
        print(f"Precision: {tp/(tp+fp):.3f}")
        print(f"Recall: {tp/(tp+fn):.3f}")
        torch.cuda.empty_cache()

        return {
            "accuracy": correct/self.dataset.__len__()*100,
            "precision": tp/(tp+fp),
            "recall": tp/(tp+fn)
        }
    
