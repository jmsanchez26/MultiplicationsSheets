import sys, json, math, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.manual_seed(0); np.random.seed(0)
d = np.load(sys.argv[1], allow_pickle=False)
xtr = torch.tensor(d['x_train'], dtype=torch.float32).div(255).unsqueeze(1); ytr = torch.tensor(d['y_train']).long()
xte = torch.tensor(d['x_test'], dtype=torch.float32).div(255).unsqueeze(1); yte = torch.tensor(d['y_test']).long()

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        s.c1 = nn.Conv2d(1, 8, 3); s.c2 = nn.Conv2d(8, 16, 3); s.f1 = nn.Linear(400, 64); s.f2 = nn.Linear(64, 10)
    def forward(s, x):
        x = F.max_pool2d(F.relu(s.c1(x)), 2)      # 28->26->13
        x = F.max_pool2d(F.relu(s.c2(x)), 2)      # 13->11->5
        x = F.relu(s.f1(x.flatten(1)))
        return s.f2(x)

def augment(x):
    n = x.shape[0]
    a = (torch.rand(n) * 2 - 1) * math.radians(16)
    sc = 1 + (torch.rand(n) * 2 - 1) * 0.16
    sh = (torch.rand(n) * 2 - 1) * 0.25
    tx = (torch.rand(n) * 2 - 1) * 0.16; ty = (torch.rand(n) * 2 - 1) * 0.16
    cos, sin = torch.cos(a), torch.sin(a)
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0] = cos / sc; theta[:, 0, 1] = (-sin + sh) / sc; theta[:, 0, 2] = tx
    theta[:, 1, 0] = sin / sc; theta[:, 1, 1] = cos / sc; theta[:, 1, 2] = ty
    g = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, g, align_corners=False)
    r = torch.rand(n, 1, 1, 1)
    thick = F.max_pool2d(x, 3, 1, 1); thin = -F.max_pool2d(-x, 3, 1, 1)
    x = torch.where(r < 0.2, thick, torch.where(r < 0.4, thin, x))
    return x

net = Net(); opt = torch.optim.Adam(net.parameters(), 2e-3)
EP = 14; sched = torch.optim.lr_scheduler.OneCycleLR(opt, 4e-3, total_steps=EP * (60000 // 256 + 1))
for ep in range(EP):
    perm = torch.randperm(60000); net.train()
    for i in range(0, 60000, 256):
        idx = perm[i:i + 256]; xb = augment(xtr[idx]); yb = ytr[idx]
        loss = F.cross_entropy(net(xb), yb); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval()
    with torch.no_grad():
        acc = (net(xte).argmax(1) == yte).float().mean().item()
        torch.manual_seed(1); aacc = (net(augment(xte)).argmax(1) == yte).float().mean().item(); torch.manual_seed(ep + 5)
    print(ep, 'loss %.3f clean %.4f aug %.4f' % (loss.item(), acc, aacc), flush=True)

def r(t, k=3): return np.round(t.detach().numpy().astype(np.float64), k).flatten().tolist()
out = {'c1w': r(net.c1.weight), 'c1b': r(net.c1.bias), 'c2w': r(net.c2.weight), 'c2b': r(net.c2.bias),
       'f1w': r(net.f1.weight), 'f1b': r(net.f1.bias), 'f2w': r(net.f2.weight), 'f2b': r(net.f2.bias)}
json.dump(out, open(sys.argv[2], 'w'), separators=(',', ':'))
# sample test vectors for JS parity check
net.eval()
with torch.no_grad():
    xs = xte[:20]; lg = net(xs)
json.dump({'x': [np.round(x.flatten().numpy(), 4).tolist() for x in xs], 'logits': np.round(lg.numpy(), 4).tolist(), 'y': yte[:20].tolist()}, open(sys.argv[2] + '.test.json', 'w'), separators=(',', ':'))
