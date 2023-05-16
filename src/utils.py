import torch
import numpy as np


def collate_batch(batch):
    targets_list = []
    images_list = []
    metadata_list = []
    for dict_ in batch:
        # append target
        targets_list.append(dict_["targets"])

        # append image after loading it
        images_list.append(dict_["image"])

        # append metadata
        metadata_list.append(dict_["metadata"])

    images = torch.stack(images_list)
    targets = torch.stack(targets_list)
    return {"metadata": metadata_list, "image": images, "targets": targets}


def cosine_similarity(x, y):
    return np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))


def recall_at_k(embeddings, labels, num_classes, k=3):
    correct_retrievals = 0
    total_queries = len(embeddings)

    for i, query_embedding in enumerate(embeddings):
        similarities = []

        for j, candidate_embedding in enumerate(embeddings):
            if i == j:  # Skip the comparison with itself
                continue

            similarity = cosine_similarity(query_embedding, candidate_embedding)
            similarities.append((similarity, labels[j]))

        # Sort by similarity in descending order
        sorted_similarities = sorted(similarities, key=lambda x: x[0], reverse=True)

        # Get the weighted sum of the top k_neighbors genre indices
        weighted_genre_indices = np.zeros(num_classes)

        for sim, genre in sorted_similarities[:k]:
            genre_index = np.argmax(genre)
            weighted_genre_indices[genre_index] += sim

        # Find the genre index with the highest score
        predicted_genre_index = np.argmax(weighted_genre_indices)

        # Check if the predicted genre index matches the query movie genre index
        if predicted_genre_index == np.argmax(labels[i]):
            correct_retrievals += 1
    return correct_retrievals / total_queries
