import torch

class MemoryBank:

    def __init__(self, size, embedding_dim, device):

        self.size = size
        self.embedding_dim = embedding_dim
        self.device = device

        self.memory = torch.zeros(size, embedding_dim, device=self.device)  # 存储正确分类的嵌入向量
        self.memory_ptr = 0  # 循环缓冲区的指针

        self.num_samples = 0  # 当前存储的样本数量

        self.class_center = None  # 类中心向量，用于动量更新

    def update(self, embeddings):

        batch_size = embeddings.size(0)
        if batch_size == 0:
            return

        embeddings = embeddings.detach()


        ptr = int(self.memory_ptr)
        if ptr + batch_size > self.size:
            num_wrap = ptr + batch_size - self.size
            self.memory[ptr:] = embeddings[:batch_size - num_wrap]
            self.memory[:num_wrap] = embeddings[batch_size - num_wrap:]
            self.memory_ptr = num_wrap
        else:
            self.memory[ptr:ptr + batch_size] = embeddings
            self.memory_ptr = ptr + batch_size

        self.num_samples = min(self.size, self.num_samples + batch_size)

    def get_class_center(self):

        if self.class_center is None:
            raise ValueError("Class center is not initialized.")
        return self.class_center

    def update_class_center(self):

        if self.num_samples == 0:
            return

        samples = self.memory[:self.num_samples]  # (num_samples, embedding_dim)
        mean_embedding = samples.mean(dim=0)  # (embedding_dim,)


        self.class_center = mean_embedding


class MemoryBankManager:

    def __init__(self, num_classes, bank_size, embedding_dim, device, min_samples_required=32):
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.device = device
        self.min_samples_required = min_samples_required  # 采样所需的最小样本数量
        self.banks = {i: MemoryBank(bank_size, embedding_dim, device) for i in range(num_classes)}

    def update(self, embeddings, labels, predicted_labels):

        correct_mask = (predicted_labels == labels)


        correct_embeddings = embeddings[correct_mask]  # (num_correct, embedding_dim)
        correct_labels = labels[correct_mask]  # (num_correct,)

        if correct_embeddings.size(0) == 0:
            return

        for class_label in torch.unique(correct_labels):
            class_mask = (correct_labels == class_label)
            class_embeddings = correct_embeddings[class_mask]

            if class_embeddings.dim() == 1:
                class_embeddings = class_embeddings.unsqueeze(0)

            self.banks[class_label.item()].update(class_embeddings)
    def get_class_centers(self):

        class_centers = {}
        for i in range(self.num_classes):
            bank = self.banks[i]
            if bank.num_samples >= self.min_samples_required and bank.class_center is not None:
                class_centers[i] = bank.get_class_center()
        return class_centers

    def update_class_centers(self):
        for i in range(self.num_classes):
            self.banks[i].update_class_center()