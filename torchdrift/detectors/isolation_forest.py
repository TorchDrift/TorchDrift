## todo: Fix


class IsolationForestDriftDetector:
    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.model.cuda()
        self.model.eval()  # Careful with test time dropout

    def fit(self, train_dataset: torch.utils.data.Dataset, batch_size=32):
        train_dl = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=False
        )
        all_outputs = []
        with torch.no_grad():
            outputs = torch.cat(
                [self.model(x.cuda()).cpu() for x, _ in tqdm(train_dl)]
            ).numpy()
            true_labels = train_dataset.y
            self.train_classes = np.unique(true_labels)
            # self.prototypes = torch.tensor(
            #    [outputs[true_labels == c].mean(0) for c in self.train_classes])
            # self.all_outs = [outputs[true_labels == c] for c in self.train_classes]
            all_outputs.append(outputs)
        all_outputs = np.concatenate(all_outputs, axis=0)
        inv_classes_dictionary = dict(
            [[v, k] for k, v in train_dataset.classes.items()]
        )
        self.train_classes = np.array(
            [inv_classes_dictionary[c] for c in self.train_classes], dtype="S"
        )
        self.od_model = IsolationForest()
        self.od_model.fit(all_outputs)

    def predict_shift(
        self, input_batch: torch.Tensor, individual_samples: bool = False
    ):
        with torch.no_grad():
            outputs = self.model(input_batch.cuda()).cpu()
        ood_score = -self.od_model.score_samples(
            outputs
        )  # we have higher == more abnormal
        if not individual_samples:
            ood_score = ood_score.mean()  # or sum?
        return ood_score
