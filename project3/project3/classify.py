import torch
import torch.nn as nn
from sklearn.metrics import classification_report

class MLP(nn.Module):
    def __init__(self, n_i, n_h, n_o):
        super(MLP, self).__init__()
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(n_i, n_h)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(n_h, n_o)
    def forward(self, input):
        return self.linear2(self.relu(self.linear1(self.flatten(input))))

# ==========load data===========
data0 = torch.load('./data.pth')
X = data0['feature']
y = data0['label']
print(f'X.shape = {X.shape}')
print(f'X.shape = {y.shape}')
X_train = X[:48000]
y_train = y[:48000]
X_test = X[48000:]
y_test = y[48000:]
# ==========create model===========
num_epochs = 1500
learning_rate = 0.01 
models = MLP(X_train.shape[1],X_train.shape[1] // 2,len(torch.unique(y_train)))
criterions = torch.nn.CrossEntropyLoss()
optimizers = torch.optim.SGD(models.parameters(), lr=learning_rate)
# ==========train model===========
for epoch in range(num_epochs):
    models.train()
    optimizers.zero_grad()
    # Forward pass
    y_pred  = models(X_train)
    # Compute Loss
    loss = criterions(y_pred, y_train)
    # Backward pass
    loss.backward()
    optimizers.step()
    if (epoch+1) % 100 == 0:                                         
        # printing loss values on every 10 epochs to keep track
        print(f'epoch: {epoch+1}, loss = {loss.item():.4f}')
# ==========evaluate model===========
with torch.no_grad():
    logits = models(X_test)
    y_pred = torch.nn.Softmax(dim=1)(logits)
    y_predicted_cls = y_pred.argmax(1)
    acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
    print(f'accuracy: {acc.item():.4f}')
    

print(classification_report(y_test, y_predicted_cls))
# ==========save model===========
torch.save(models.state_dict(),'./Classify_Model.pth')

# ==========load model===========
# modelA = MLP(X_train.shape[1],X_train.shape[1] // 2,len(torch.unique(y_train)))
# modelA.load_state_dict(torch.load('./Classify_Model.pth'))
# modelA.eval()
# with torch.no_grad():
#     logits = modelA(X_test)
#     y_pred = torch.nn.Softmax(dim=1)(logits)
#     y_predicted_cls = y_pred.argmax(1)
#     acc = y_predicted_cls.eq(y_test).sum() / float(y_test.shape[0])
#     print(f'accuracy: {acc.item():.4f}')
# from sklearn.metrics import classification_report
# print(classification_report(y_test, y_predicted_cls))