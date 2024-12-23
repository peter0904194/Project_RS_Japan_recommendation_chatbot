import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import MakeSequenceDataSet, BERTRecDataSet
from model import BERT
from tqdm import tqdm
from train_and_eval import train, evaluate
import pandas as pd 


def main():
    config = {
        'data_path': '/Users/mac/AIworkspace/Bitamin/bert4rec/',
        'max_len': 50,
        'hidden_units': 50,  # Embedding
        'num_heads': 1,  # Multi-head layer
        'num_layers': 2,  # block encoder layer
        'dropout_rate': 0.5,  # dropout
        'lr': 0.001,
        'batch_size': 128,
        'num_epochs': 50,
        'num_workers': 2,
        'mask_prob': 0.15,
    }

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    make_sequence_dataset = MakeSequenceDataSet(config['data_path'])
    user_train, movie_genres, user_valid = make_sequence_dataset.get_train_valid_data()

    bert4rec_dataset = BERTRecDataSet(
        user_train=user_train,
        movie_genres=movie_genres,
        max_len=config['max_len'],
        num_user=make_sequence_dataset.num_user,
        num_item=make_sequence_dataset.num_item,
        mask_prob=config['mask_prob'],
    )

    data_loader = DataLoader(
        bert4rec_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=config['num_workers'],
    )

    model = BERT(
        num_items=make_sequence_dataset.num_item,
        genres_size=18,
        bert_hidden_units=config['hidden_units'],
        bert_num_heads=config['num_heads'],
        bert_num_blocks=config['num_layers'],
        bert_max_len=config['max_len'],
        bert_dropout=config['dropout_rate'],
    ).to(device)

    # criterion = nn.CrossEntropyLoss(ignore_index=0)
    # optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])

    # loss_list = []
    # for epoch in tqdm(range(1, config['num_epochs'] + 1)):
    #     train_loss = train(
    #         model=model,
    #         criterion=criterion,
    #         optimizer=optimizer,
    #         data_loader=data_loader,
    #         device=device
    #     )
    #     loss_list.append(train_loss)

    #     print(f'Epoch: {epoch:3d}| Train loss: {train_loss:.5f}')

    # torch.save(model.state_dict(), 'weights.pth')

    model.load_state_dict(torch.load('weights.pth', weights_only=True))
    
    model.eval()
   
    # 결과를 저장할 리스트 초기화
    results = []
    num_item_sample = 100
    users = [user for user in range(make_sequence_dataset.num_user)]

    for user in tqdm(users):
        seq = (user_train[user] + [make_sequence_dataset.num_item + 1])[-config['max_len']:]
        genre_seq = ([make_sequence_dataset.movie_genres(i) for i in user_train[user]] + [[1]*18])[-config['max_len']:]
        padding_len = config['max_len'] - len(seq)

        seq = [0] * padding_len + seq
        genre_seq = [[0]*18] * padding_len + genre_seq

        rated = user_train[user] + [user_valid[user]]
        actual_item_id = user_valid[user]
        neg_samples = bert4rec_dataset.random_neg_sampling(rated_item=rated, num_item_sample=num_item_sample)
        items = [actual_item_id] + neg_samples

        with torch.no_grad():
            genre_seq = torch.Tensor([genre_seq]).to(device)
            seq = torch.LongTensor([seq]).to(device)

            # 모델 예측 수행
            predictions = model(seq, genre_seq)
            predictions = predictions[0][-1][items]

            # 실제 아이템 점수 및 순위 확인
            actual_item_score = predictions[0].item()
            sorted_predictions = predictions.argsort(descending=True)
            actual_item_rank = (sorted_predictions == 0).nonzero(as_tuple=True)[0].item()

            # 점수가 높은 예측 아이템 및 순위 기록
            best_prediction_idx = predictions.argmax().item()
            best_item_id = items[best_prediction_idx]
            best_score = predictions[best_prediction_idx].item()

            # 결과를 리스트에 저장
            results.append({
                "user_id": user,
                "actual_item_id": actual_item_id,
                "actual_item_rank": actual_item_rank,
                "predicted_item_id": best_item_id,
                "predicted_score": best_score
            })

    # DataFrame으로 변환 후 CSV로 저장
    df = pd.DataFrame(results)
    df.to_csv("user_last_item_predictions_with_items_recent.csv", index=False)

    print("CSV 파일로 저장이 완료되었습니다.")

if __name__ == "__main__":
    main()