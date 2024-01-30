import torch

def contrastive_loss(graph_embedding, text_embedding, normalize=False):
    """
    graph_embedding: (batch_size, embedding_size)
    text_embedding: (batch_size, embedding_size)
    """
    if normalize:
        graph_embedding = torch.nn.functional.normalize(graph_embedding, p=2, dim=1)
        text_embedding = torch.nn.functional.normalize(text_embedding, p=2, dim=1)
    logits = torch.matmul(graph_embedding, torch.transpose(text_embedding, 0, 1))
    labels = torch.arange(logits.shape[0], device=graph_embedding.device)
    return torch.nn.functional.cross_entropy(logits, labels) + torch.nn.functional.cross_entropy(torch.transpose(logits, 0, 1), labels)

def self_supervised_entropy(student_embedding, teacher_embedding, center, temperature_student, temperature_teacher):
    """
    student_embedding: (batch_size, embedding_size)
    teacher_embedding: (batch_size, embedding_size)
    center: (embedding_size)
    """
    print(student_embedding.shape, teacher_embedding.shape, center.shape)
    print(temperature_student, temperature_teacher)
    s = torch.nn.functional.softmax(student_embedding / temperature_student, dim=1)
    t = torch.nn.functional.softmax((teacher_embedding - center) / temperature_teacher, dim=1)
    return torch.mean(torch.sum(-s * torch.log(t), dim=1))
