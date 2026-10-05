import math
import torch
from torch import nn
import torch.nn.functional as F

class ADNM_km(nn.Module):
    def __init__(self, input_size, out_size, M=5, device='cpu'):
        super(ADNM_km, self).__init__()

        self.input_size = input_size
        w = torch.rand([out_size, M, input_size]).to(device)
        q = torch.rand([out_size, M, input_size]).to(device)
        m = torch.rand([out_size, 1, input_size]).to(device)
        D_k = torch.rand([out_size, M]).to(device)
        D_q = torch.rand([out_size, M]).to(device)
        torch.nn.init.constant_(q, 0.1)
        torch.nn.init.uniform_(m, a=-10.0, b=10.0)
        
        self.params = nn.ParameterDict({'w': nn.Parameter(w)})
        self.params.update({'q': nn.Parameter(q)})
        self.params.update({'m': nn.Parameter(m)})
        self.params.update({'D_k': nn.Parameter(D_k)})
        self.params.update({'D_q': nn.Parameter(D_q)})

    def forward(self, x):
        # Synapse
        out_size, M, _ = self.params['w'].shape
        size_dims = x.size()
        x = torch.unsqueeze(x, -2)
        x = torch.unsqueeze(x, -3)
        # print("x_unsqueeze:", x.shape)
        # x = x.repeat(-1, out_size, M, 1)
        # print("x_repeat:", x.shape)
        x = x.expand(*size_dims[:-1], out_size, M, *size_dims[-1:])
        # print("x_expand:", x.shape)
        S = torch.sigmoid(torch.mul(x, self.params['w']) - self.params['q'])
        # print("S:", S.shape)

        A = torch.tanh(self.params['m'])       
        # Dendritic
        D = torch.sum(torch.mul(S, A), -1) 
        # print("D:", D.shape)

        D = self.params['D_k'] * D - self.params['D_q']

        # Membrane Soma
        O = torch.sum(torch.sigmoid(D), -1)

        return O
    
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)


class ADNM_ln(nn.Module):
    def __init__(self, input_size, out_size, M=5, device='cpu'):
        super(ADNM_ln, self).__init__()

        self.input_size = input_size
        w = torch.rand([out_size, M, input_size]).to(device)
        q = torch.rand([out_size, M, input_size]).to(device)
        m = torch.rand([out_size, 1, input_size]).to(device)
        D_k = torch.rand([out_size, M]).to(device)
        D_q = torch.rand([out_size, M]).to(device)
        torch.nn.init.constant_(q, 0.1)
        torch.nn.init.uniform_(m, a=-10.0, b=10.0)
        
        self.params = nn.ParameterDict({'w': nn.Parameter(w)})
        self.params.update({'q': nn.Parameter(q)})
        self.params.update({'m': nn.Parameter(m)})
        self.params.update({'D_k': nn.Parameter(D_k)})
        self.params.update({'D_q': nn.Parameter(D_q)})

        self.ln1 = nn.LayerNorm(M)

    def forward(self, x):
        # Synapse
        out_size, M, _ = self.params['w'].shape
        
        size_dims = x.size()
        x = torch.unsqueeze(x, -2)
        x = torch.unsqueeze(x, -3)
        # print("x_unsqueeze:", x.shape)
        # x = x.repeat(-1, out_size, M, 1)
        # print("x_repeat:", x.shape)
        x = x.expand(*size_dims[:-1], out_size, M, *size_dims[-1:])
        # print("x_expand:", x.shape)
        S = torch.sigmoid(torch.mul(x, self.params['w']) - self.params['q'])
        # print("S:", S.shape)
        
        A = torch.tanh(self.params['m'])       
        # Dendritic
        D = torch.sum(torch.mul(S, A), -1) 
        # print("D:", D.shape)
        
        D = self.params['D_k'] * D - self.params['D_q']
        D = self.ln1(D)

        # Membrane Soma
        O = torch.sum(torch.sigmoid(D), -1)

        return O
    
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

class ADNM_new(nn.Module):
    def __init__(self, input_size, out_size, M):
        super(ADNM_new, self).__init__()
        DNM_W = torch.rand([input_size, 1, out_size * M])
        q = torch.rand([input_size, 1, out_size * M])
        m = torch.rand([out_size, 1, input_size])
        D_k = torch.rand([out_size, M])
        D_q = torch.rand([out_size, M])
        torch.nn.init.constant_(q, 0.1)
        torch.nn.init.uniform_(m, a=-10.0, b=10.0)
        self.params = nn.ParameterDict({'DNM_W': nn.Parameter(DNM_W),
                                        'q': nn.Parameter(q),
                                        'm': nn.Parameter(m),
                                        'D_k': nn.Parameter(D_k),
                                        'D_q': nn.Parameter(D_q)})
        self.input_size = input_size
        self.out_size = out_size
        self.M = M
    
    def forward(self, x):
        x = x.transpose(0, 1).unsqueeze(2)  # [input_size, batch, 1]
        x = torch.matmul(x, self.params['DNM_W']) - self.params['q']
        x = x.view(self.input_size, x.size(1), self.M, self.out_size).permute(1, 3, 2, 0)
        x = torch.sigmoid(x)
        A = torch.tanh(self.params['m'])
        x = torch.mul(x, A)
        x = torch.sum(x, 3)
        x = self.params['D_k'] * x - self.params['D_q']
        x = torch.sigmoid(x)
        x = torch.sum(x, 2)
        return x

class ADNM_new2(nn.Module):
    def __init__(self, input_size, out_size, M, device='cpu'):
        super(ADNM_new2, self).__init__()
        
        w = torch.rand([out_size, M, input_size]).to(device)
        q = torch.rand([out_size, M, input_size]).to(device)
        m = torch.rand([out_size, 1, input_size]).to(device)
        D_k = torch.rand([out_size, M]).to(device)
        D_q = torch.rand([out_size, M]).to(device)

        torch.nn.init.constant_(q, 0.1)  
        torch.nn.init.uniform_(m, a=-10.0, b=10.0)

        self.params = nn.ParameterDict({
            'w': nn.Parameter(w),
            'q': nn.Parameter(q),
            'm': nn.Parameter(m),
            'D_k': nn.Parameter(D_k),
            'D_q': nn.Parameter(D_q)
        })
    
    def forward(self, x):

        x = x / x.norm(dim=-1, keepdim=True)  # normalise input

        # Synapse
        S = torch.sigmoid(torch.einsum('bi,omi->bomi', x, self.params['w']) - self.params['q'])
        
        # Dendritic
        A = torch.tanh(self.params['m'])
        D = torch.sum(S * A, -1)  # [b, out_size, M]

        # Membrane
        D = self.params['D_k'] * D - self.params['D_q']
        
        # Soma
        O = torch.sum(torch.sigmoid(D), -1)
        return O


    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

class ADNM_ifm(nn.Module):
    def __init__(self, input_size, out_size, M, device='cpu'):
        super(ADNM_ifm, self).__init__()
        self.M = M
        self.out_size = out_size

        m = torch.rand([M]).to(device)

        torch.nn.init.uniform_(m, a=-10.0, b=10.0)

        self.params = nn.ParameterDict({
            'm': nn.Parameter(m),
        })

        self.Linears = nn.ModuleList()
        for i in range(M):
            self.Linears.append(nn.Linear(input_size, out_size))
    
    def forward(self, x):

        # x = x / x.norm(dim=-1, keepdim=True)
        output = torch.zeros([x.size(0), self.out_size, self.M],
                                          dtype=x.dtype).to(x.device)
        for i in range(self.M):
            output[:, :, i] = self.Linears[i](x)  # [b, out_size, M]
        output = torch.sigmoid(output)
        
        # Dendritic
        A = torch.tanh(self.params['m'])
        O = torch.sum(output * A, -1)  # [b, out_size]
        
        return O

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

class ADNM_ifm2(nn.Module):
    def __init__(self, input_size, out_size, M, device='cpu'):
        super(ADNM_ifm2, self).__init__()
        self.M = M
        self.out_size = out_size

        m = torch.rand([M]).to(device)

        torch.nn.init.uniform_(m, a=-10.0, b=10.0)

        self.params = nn.ParameterDict({
            'm': nn.Parameter(m),
        })

        # self.Linears = nn.ModuleList()
        # for i in range(M):
        #     self.Linears.append(nn.Linear(input_size, out_size))

        self.mlp = nn.Linear(input_size, out_size * M)

    def forward(self, x):

        # x = x / x.norm(dim=-1, keepdim=True)
        # output = torch.zeros([x.size(0), self.out_size, self.M],
        #                                   dtype=x.dtype).to(x.device)
        # for i in range(self.M):
        #     output[:, :, i] = self.Linears[i](x)  # [b, out_size, M]
        # output = torch.sigmoid(output)
        x = self.mlp(x)
        x = torch.sigmoid(x)
        x = x.reshape(x.size(0), self.out_size, self.M)
        
        A = torch.tanh(self.params['m'])
        O = torch.sum(x * A, -1)  
        
        return O

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)


class ADNM_ifm3(nn.Module):
    def __init__(self, input_size, out_size, M, device='cpu'):
        super(ADNM_ifm3, self).__init__()
        self.M = M
        self.input_size = input_size
        self.out_size = out_size

        m = torch.rand([out_size, M]).to(device)

        torch.nn.init.uniform_(m, a=-10.0, b=10.0)

        self.params = nn.ParameterDict({
            'm': nn.Parameter(m),
        })

        self.mlp = nn.Linear(input_size, out_size * M)
        self.act = nn.Sigmoid()
        # self.dendritic = nn.Linear(M, 1)

    def forward(self, x):
        x = self.mlp(x)
        x = self.act(x)
        x = x.view(*x.shape[:-1], self.out_size, self.M)

        A = torch.tanh(self.params['m'])
        O = torch.sum(x * A, -1) 
        
        return O

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)



class DNM(nn.Module):
    def __init__(self, input_size, out_size, M, device='cpu'):
        super(DNM, self).__init__()
        self.M = M
        self.input_size = input_size
        self.out_size = out_size

        m = torch.rand([out_size, M]).to(device)
        torch.nn.init.uniform_(m, a=-10.0, b=10.0)

        self.params = nn.ParameterDict({
            'm': nn.Parameter(m),
        })

        self.mlp = nn.Linear(input_size, out_size * M)
        self.act = nn.Sigmoid()

    def forward(self, x):
        x = self.mlp(x)
        x = self.act(x)
        x = x.view(*x.shape[:-1], self.out_size, self.M)

        A = torch.tanh(self.params['m'])
        O = torch.sum(x * A, -1)
        return O

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

class DNM_gate2_linear(nn.Module): 
    def __init__(self, input_size, out_size, M=5, device='cpu'):
        super(DNM_gate2_linear, self).__init__()


        self.input_size = input_size

        w = torch.rand([out_size, M, input_size]).to(device)

        q = torch.full([out_size, M, input_size], 0.1).to(device)

        D_k = torch.rand([out_size, M]).to(device)

        D_q = torch.rand([out_size, M]).to(device)
        
        self.params = nn.ParameterDict({
            'w': nn.Parameter(w),
            'q': nn.Parameter(q),
            'D_k': nn.Parameter(D_k),
            'D_q': nn.Parameter(D_q)
        })

    def forward(self, x):
        out_size, M, _ = self.params['w'].shape
        size_dims = x.size()
        x = x.unsqueeze(-2).unsqueeze(-3)
        x = x.expand(*size_dims[:-1], out_size, M, *size_dims[-1:])

        raw_S = torch.mul(x, self.params['w']) - self.params['q']
        S = torch.sigmoid(raw_S)

        total_response = torch.sum(S, dim=-1, keepdim=True)
        
        enhanced_S = S / (total_response + 1e-6) 
        S = S * enhanced_S  
        
        D = torch.sum(S, -1)

        adaptive_D_k = self.params['D_k'] + torch.mean(S, dim=-1)
        adaptive_D_q = self.params['D_q'] + torch.mean(S, dim=-1)

        D = adaptive_D_k * D - adaptive_D_q

        O = torch.sum(torch.sigmoid(D), -1)

        return O

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)


class ADNM_ifm4(nn.Module):
    def __init__(self, input_size, out_size, M, device='cpu'):
        super(ADNM_ifm4, self).__init__()
        self.M = M
        self.input_size = input_size
        self.out_size = out_size

        m = torch.rand([out_size, M]).to(device)

        torch.nn.init.uniform_(m, a=-10.0, b=10.0)

        self.params = nn.ParameterDict({
            'm': nn.Parameter(m),
        })

        # self.Linears = nn.ModuleList()
        # for i in range(M):
        #     self.Linears.append(nn.Linear(input_size, out_size))

        self.mlp = nn.Linear(input_size, out_size * M)
        self.act = nn.Sigmoid()
        self.dendritic = nn.Linear(out_size * M, out_size)

    def forward(self, x):

        # x = x / x.norm(dim=-1, keepdim=True)
        # output = torch.zeros([x.size(0), self.out_size, self.M],
        #                                   dtype=x.dtype).to(x.device)
        # for i in range(self.M):
        #     output[:, :, i] = self.Linears[i](x)  # [b, out_size, M]
        # output = torch.sigmoid(output)
        x = self.mlp(x)
        x = self.act(x)
        # x = x.reshape(x.size(0), self.out_size, self.M)
        
        # A = torch.tanh(self.params['m'])
        # O = torch.sum(x * A, -1)  
        # D = D.reshape(x.size(0), -1)
        O = self.dendritic(x)#.squeeze(-1)
        
        return O

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

class ADNM_ifm5(nn.Module):
    def __init__(self, input_size, out_size, M, device='cpu'):
        super(ADNM_ifm5, self).__init__()
        self.M = M
        self.input_size = input_size
        self.out_size = out_size

        m = torch.rand([out_size, input_size]).to(device)

        torch.nn.init.uniform_(m, a=-10.0, b=10.0)

        self.params = nn.ParameterDict({
            'm': nn.Parameter(m),
        })

        # self.Linears = nn.ModuleList()
        # for i in range(M):
        #     self.Linears.append(nn.Linear(input_size, out_size))

        self.mlp = nn.Linear(input_size, out_size * input_size)
        self.act = nn.Sigmoid()
        # self.dendritic = nn.Linear(out_size * M, out_size)

    def forward(self, x):

        # x = x / x.norm(dim=-1, keepdim=True) 
        # output = torch.zeros([x.size(0), self.out_size, self.M],
        #                                   dtype=x.dtype).to(x.device)
        # for i in range(self.M):
        #     output[:, :, i] = self.Linears[i](x)  # [b, out_size, M]
        # output = torch.sigmoid(output)
        x = self.mlp(x)
        x = self.act(x)
        x = x.reshape(x.size(0), self.out_size, self.input_size)
        
        A = torch.tanh(self.params['m'])
        O = torch.sum(x * A, -1) 
        # D = D.reshape(x.size(0), -1)
        # O = self.dendritic(x)#.squeeze(-1)
        
        return O

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)


class ADNM_easy(nn.Module):
    def __init__(self, input_size, out_size, M, device='cpu'):
        super(ADNM_easy, self).__init__()
        
        w = torch.rand([out_size, input_size]).to(device)
        q = torch.rand([out_size, input_size]).to(device)
        m = torch.rand([input_size]).to(device)
        torch.nn.init.constant_(q, 0.1)  
        torch.nn.init.uniform_(m, a=-10.0, b=10.0)

        self.params = nn.ParameterDict({
            'w': nn.Parameter(w),
            'q': nn.Parameter(q),
            'm': nn.Parameter(m),
        })
    
    def forward(self, x):

        # x = x / x.norm(dim=-1, keepdim=True)

        S = torch.tanh(torch.einsum('bi,oi->boi', x, self.params['w']) - self.params['q'])
        
        A = torch.tanh(self.params['m'])
        D = torch.sum(S * A, -1) 
        

        O = D
        return O


    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)


class ADNM_easy2(nn.Module):
    def __init__(self, input_size, out_size, M, device='cpu'):
        super(ADNM_easy2, self).__init__()
        
        w = torch.rand([out_size, input_size]).to(device)
        q = torch.rand([out_size, input_size]).to(device)
        m = torch.rand([M]).to(device)

        torch.nn.init.constant_(q, 0.1) 
        torch.nn.init.uniform_(m, a=-10.0, b=10.0)


        self.params = nn.ParameterDict({
            'w': nn.Parameter(w),
            'q': nn.Parameter(q),
            'm': nn.Parameter(m),
        })
        self.in2m = nn.Linear(input_size, M)
    
    def forward(self, x):

        # x = x / x.norm(dim=-1, keepdim=True) 

        S = torch.tanh(torch.einsum('bi,oi->boi', x, self.params['w']) - self.params['q'])

        D = self.in2m(S)  # [b, o, m]
        
        A = torch.tanh(self.params['m'])
        O = torch.sum(D * A, -1)  
        
        return O


    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

class ADNS_m1(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, M=5):
        super(ADNS_m1, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.DNM_Linear1 = ADNM_easy2(input_size, hidden_size, M)
        self.DNM_Linear2 = ADNM_easy2(hidden_size, hidden_size, M)
        self.DNM_Linear3 = ADNM_easy2(hidden_size, out_size, M)
    
    def forward(self, x):
        # x = x.view(-1, self.input_size)
        x = self.DNM_Linear1(x)
        x = self.DNM_Linear2(x)
        out = self.DNM_Linear3(x)
        return out
    
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

class ADNS_km(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, M=5):
        super(ADNS_km, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.DNM_Linear1 = ADNM_km(input_size, hidden_size, M)
        self.DNM_Linear2 = ADNM_km(hidden_size, out_size, M)
    
    def forward(self, x):
        # x = x.view(-1, self.input_size)
        x = self.DNM_Linear1(x)
        out = self.DNM_Linear2(x)
        return out
    
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

class ADNS_ln(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, M=5):
        super(ADNS_ln, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.DNM_Linear1 = ADNM_ln(input_size, hidden_size, M)
        self.DNM_Linear2 = ADNM_ln(hidden_size, out_size, M)
    
    def forward(self, x):
        # x = x.view(-1, self.input_size)
        x = self.DNM_Linear1(x)
        out = self.DNM_Linear2(x)
        return out
    
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

class ADNS_3(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, M=5):
        super(ADNS_3, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.ifm = IFM(input_size, M, 0.1)
        self.DNM_Linear1 = ADNM_ln(input_size, hidden_size, M)
        self.DNM_Linear2 = ADNM_ln(hidden_size, hidden_size, M)
        # self.DNM_Linear3 = ADNM_ifm4(hidden_size, out_size, M)
        self.DNM_Linear3 = ADNM_ln(hidden_size, out_size, M)
        self.act = nn.ReLU()
    
    def forward(self, x):
        # x = x.view(-1, self.input_size)
        # x = self.ifm(x)
        x = self.DNM_Linear1(x)
        x_res = self.act(x)
        x = self.DNM_Linear2(x_res) + x_res
        x = self.act(x)
        out = self.DNM_Linear3(x)
        return out
    
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

class ADNS_IFM(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, M=5):
        super(ADNS_IFM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # self.ifm = IFM(input_size, M, 0.1)
        self.DNM_Linear1 = ADNM_ifm3(input_size, hidden_size, M)
        # self.DNM_Linear2 = ADNM_ifm3(hidden_size, hidden_size, M)
        # self.DNM_Linear3 = ADNM_ifm4(hidden_size, out_size, M)
        self.DNM_Linear3 = ADNM_ifm3(hidden_size, out_size, M)
        self.act = nn.ReLU()
    
    def forward(self, x):
        # x = x.view(-1, self.input_size)
        # x = self.ifm(x)
        x = self.DNM_Linear1(x)
        # x_res = self.act(x)
        # x = self.DNM_Linear2(x_res) + x_res
        x = self.act(x)
        out = self.DNM_Linear3(x)
        return out
    
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)


class ADNS_IFM2(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, M=5):
        super(ADNS_IFM2, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.ifm = IFM(input_size, M, 0.1)
        self.DNM_Linear1 = ADNM_ifm3(input_size*M*2, hidden_size, M)
        self.DNM_Linear2 = ADNM_ifm3(hidden_size, hidden_size, M)
        # self.DNM_Linear3 = ADNM_ifm4(hidden_size, out_size, M)
        self.DNM_Linear3 = ADNM_ifm3(hidden_size, out_size, M)
        self.act = nn.ReLU()
    
    def forward(self, x):
        # x = x.view(-1, self.input_size)
        x = self.ifm(x)
        x = self.DNM_Linear1(x)
        x_res = self.act(x)
        x = self.DNM_Linear2(x_res) + x_res
        x = self.act(x)
        out = self.DNM_Linear3(x)
        return out
    
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

class D_model(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, M=5):
        super(D_model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        self.ifm = IFM(input_size, M, 0.1)
        self.mlp = nn.Sequential(
            nn.Linear(input_size * 2 * M, hidden_size),
            nn.ReLU()
        )
                                 
        self.DNM_Linear = ADNM_easy(input_size * 2 * M, hidden_size, M)
        # self.mlp = ADNM_easy(hidden_size, hidden_size, M)
        # self.mlp2 = ADNM_easy(hidden_size, hidden_size, M)
        # self.mlp = nn.Sequential(
        #     nn.Linear(hidden_size, hidden_size*4),
        #     nn.ReLU(),
        #     nn.Dropout(),
        #     nn.Linear(hidden_size*4, hidden_size),
        #     nn.ReLU(),
        # )
        self.out_Linear = ADNM_km(hidden_size, out_size, M)
    
    def forward(self, x):
        x = self.ifm(x)
        # x = self.DNM_Linear(x)
        x = self.mlp(x)
        # x = self.mlp(x) + self.mlp2(x)
        # x = self.mlp2(x)
        out = self.out_Linear(x)
        return out
    
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

class DNM_Linear(nn.Module):    # MDNN
    def __init__(self, input_size, out_size, M=5, device='cpu'):
        super(DNM_Linear, self).__init__()

        Synapse_W = torch.rand([out_size, M, input_size]).to(device)#.cuda() # [size_out, M, size_in]
        Synapse_q = torch.rand([out_size, M, input_size]).to(device)#.cuda()
        torch.nn.init.constant_(Synapse_q, 0.1)
        k = torch.rand(1).to(device)
        qs = torch.rand(1).to(device)

        self.params = nn.ParameterDict({'Synapse_W': nn.Parameter(Synapse_W)})
        self.params.update({'Synapse_q': nn.Parameter(Synapse_q)})
        self.params.update({'k': nn.Parameter(k)})
        self.params.update({'qs': nn.Parameter(qs)})
        self.input_size = input_size

    def forward(self, x):
        # Synapse
        out_size, M, _ = self.params['Synapse_W'].shape
        x = torch.unsqueeze(x, 1)
        x = torch.unsqueeze(x, 2)
        x = x.repeat(1, out_size, M, 1)
        x = 5 * torch.mul(x, self.params['Synapse_W']) - self.params['Synapse_q']
        x = torch.sigmoid(x)

        # Dendritic
        x = torch.prod(x, 3) #prod 

        # Membrane
        x = torch.sum(x, 2)

        # Soma
        x = 5 * (x - 0.5)

        return x

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

class DNM_conv(nn.Module):    # MDNN
    def __init__(self, input_size, out_size, M=5, device='cpu'):
        super(DNM_conv, self).__init__()
        self.input_size = input_size
        self.out_size = out_size
        self.M = M

        self.dendritic_layer = nn.Conv1d(in_channels=1, out_channels=out_size * M, kernel_size=input_size)
        self.soma_layer = nn.Linear(out_size * M, out_size)

    def forward(self, x):
        # Dendritic
        x = x.unsqueeze(-2)
        x = self.dendritic_layer(x)
        x = x.squeeze(-1)

        x = torch.sigmoid(x)

        # Soma
        x = self.soma_layer(x)

        return x


class DNM_multiple(nn.Module):   # MDNN * 2
    def __init__(self, input_size, hidden_size, out_size, M=5):
        super(DNM_multiple, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.DNM_Linear1 = DNM_Linear(input_size, hidden_size, M)
        self.DNM_Linear2 = DNM_Linear(hidden_size, out_size, M)
    
    def forward(self, x):
        x = x.view(-1, self.input_size)
        x = self.DNM_Linear1(x)
        out = self.DNM_Linear2(x)
        return out
    
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)


class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(MLP, self).__init__()
        self.hidden_size = hidden_size
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, out_size)
        self.drop_out = torch.nn.Dropout(0.5)

    def forward(self, x):
        x = x.float()
        x = self.l1(x)
        x = torch.relu(x)
        x = self.drop_out(x)
        x = self.l2(x)
        return x

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

class MLP_G(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(MLP_G, self).__init__()
        self.hidden_size = hidden_size
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, out_size)
        self.drop_out = torch.nn.Dropout(0.5)

    def forward(self, batch):
        batch_size = batch.batch.max().item() + 1
        x = batch.fp.view(batch_size, -1)  # [batch_size, input_size]
        x = x.float()
        x = self.l1(x)
        x = torch.relu(x)
        x = self.drop_out(x)
        x = self.l2(x)
        return x

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)



class ADNS_IFM_G(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, M=5):
        super(ADNS_IFM_G, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.DNM_Linear1 = ADNM_ifm3(input_size, hidden_size, M)
        self.DNM_Linear2 = ADNM_ifm3(hidden_size, out_size, M)
        self.act = nn.ReLU()
        self.drop_out = torch.nn.Dropout(0.5)
    
    def forward(self, batch):
        batch_size = batch.batch.max().item() + 1
        x = batch.fp.view(batch_size, -1)  # [batch_size, input_size]
        x = x.float()
        x = self.DNM_Linear1(x)
        x = self.act(x)
        out = self.DNM_Linear2(x)
        return out
    
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

class MLP_Gv3(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(MLP_Gv3, self).__init__()
        self.hidden_size = hidden_size
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, hidden_size)
        self.l3 = torch.nn.Linear(hidden_size, out_size)
        # self.drop_out = torch.nn.Dropout(0.5)

    def forward(self, batch):
        batch_size = batch.batch.max().item() + 1
        x = batch.fp.view(batch_size, -1)  # [batch_size, input_size]
        x = x.float()
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        x = torch.relu(x)
        x = self.l3(x)
        # x = self.drop_out(x)
        return x

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

class MLP_v3(nn.Module):
    def __init__(self, input_size, hidden_size, out_size):
        super(MLP_v3, self).__init__()
        self.hidden_size = hidden_size
        self.l1 = torch.nn.Linear(input_size, hidden_size)
        self.l2 = torch.nn.Linear(hidden_size, hidden_size)
        self.l3 = torch.nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = x.float()
        x = self.l1(x)
        x = torch.relu(x)
        x = self.l2(x)
        x = torch.relu(x)
        x = self.l3(x)
        return x

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)


# Adapted from efficient-kan (https://github.com/Blealtan/efficient-kan). In its docstrings,
# "the paper" and "the authors" refer to KAN (Liu et al., 2024).
class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            (
                torch.arange(-spline_order, grid_size + spline_order + 1) * h
                + grid_range[0]
            )
            .expand(in_features, -1)
            .contiguous()
        )
        self.register_buffer("grid", grid)

        self.base_weight = torch.nn.Parameter(torch.Tensor(out_features, in_features))
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = torch.nn.Parameter(
                torch.Tensor(out_features, in_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            noise = (
                (
                    torch.rand(self.grid_size + 1, self.in_features, self.out_features)
                    - 1 / 2
                )
                * self.scale_noise
                / self.grid_size
            )
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                # torch.nn.init.constant_(self.spline_scaler, self.scale_spline)
                torch.nn.init.kaiming_uniform_(self.spline_scaler, a=math.sqrt(5) * self.scale_spline)

    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        assert y.size() == (x.size(0), self.in_features, self.out_features)

        A = self.b_splines(x).transpose(
            0, 1
        )  # (in_features, batch_size, grid_size + spline_order)
        B = y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = torch.linalg.lstsq(
            A, B
        ).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(
            2, 0, 1
        )  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (
            self.out_features,
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1)
            if self.enable_standalone_scale_spline
            else 1.0
        )

    def forward(self, x: torch.Tensor):
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.view(-1, self.in_features)

        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        output = base_output + spline_output
        
        output = output.view(*original_shape[:-1], self.out_features)
        return output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(
            1, 0, 2
        )  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = torch.sort(x, dim=0)[0]
        grid_adaptive = x_sorted[
            torch.linspace(
                0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device
            )
        ]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(
                self.grid_size + 1, dtype=torch.float32, device=x.device
            ).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step
                * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step
                * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )



class KanDNM(nn.Module):
    def __init__(self, input_size, out_size, M=5, device='cpu'):
        super(KanDNM, self).__init__()
        grid_size=5
        spline_order=3
        scale_noise=0.1
        scale_base=1.0
        scale_spline=1.0
        base_activation=torch.nn.SiLU
        grid_eps=0.02
        grid_range=[-1, 1]

        self.input_size = input_size
        self.out_size = out_size
        self.M = M
        self.kan = KANLinear(
            input_size,
            M * out_size,
            grid_size=grid_size,
            spline_order=spline_order,
            scale_noise=scale_noise,
            scale_base=scale_base,
            scale_spline=scale_spline,
            base_activation=base_activation,
            grid_eps=grid_eps,
            grid_range=grid_range,
        )
        D_k = torch.rand([out_size, M]).to(device)
        D_q = torch.rand([out_size, M]).to(device)
        
        self.params = nn.ParameterDict({'D_k': nn.Parameter(D_k)})
        self.params.update({'D_q': nn.Parameter(D_q)})

    def forward(self, x):
        batch_size = x.size(0)
        # Synapse

        x = self.kan(x)
        # print("x:", x.shape)
        x = x.reshape(batch_size, self.out_size, self.M)
        # print("x:", x.shape)
    
        # Dendritic
        D = self.params['D_k'] * x - self.params['D_q']

        # Membrane Soma
        O = torch.sum(torch.sigmoid(D), -1)

        return O
    
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

class kanDNS(nn.Module):
    def __init__(self, input_size, hidden_size, out_size, M=5):
        super(kanDNS, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.DNM_Linear1 = KanDNM(input_size, hidden_size, M)
        self.DNM_Linear2 = KanDNM(hidden_size, out_size, M)
    
    def forward(self, x):
        # x = x.view(-1, self.input_size)
        x = self.DNM_Linear1(x)
        out = self.DNM_Linear2(x)
        return out
    
    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

# model = kanDNS(10, 20, 10)
# input_tensor = torch.zeros(32, 10)

# out_tensot = model(input_tensor)
# print(out_tensot.shape)



class IFM(nn.Module):
    def __init__(self, input_dim, k=10, sigma=0.1):
        """
        Independent Feature Mapping (IFM) Layer
        Args:
            input_dim: Dimensionality of the input features
            k: Number of frequency components
            sigma: Standard deviation for initializing the parameters c
        """
        super(IFM, self).__init__()
        self.k = k
        self.c = nn.Parameter(torch.randn(input_dim, k) * sigma)
    
    def forward(self, x):
        """
        Forward pass of IFM.
        Args:
            x: Input tensor of shape (batch_size, input_dim)
        Returns:
            Transformed tensor of shape (batch_size, input_dim * 2 * k)
        """
        # Compute v = 2π * c * x
        v = 2 * torch.pi * x.unsqueeze(-1) * self.c  # Shape: (batch_size, input_dim, k)
        # Compute [sin(v), cos(v)]
        sin_v = torch.sin(v)
        cos_v = torch.cos(v)
        return torch.cat([sin_v, cos_v], dim=-1).view(x.size(0), -1)  # Flatten to (batch_size, input_dim * 2 * k)

class IFMMLP_G(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, k=8, sigma=6):
        """
        MLP with IFM layer
        Args:
            input_dim: Original input dimensionality
            hidden_dim: Number of hidden units in the MLP
            output_dim: Output dimensionality
            k: Number of frequency components for IFM
            sigma: Standard deviation for IFM parameters
        """
        super(IFMMLP_G, self).__init__()
        self.ifm = IFM(input_dim, k, sigma)
        self.fc1 = nn.Linear(input_dim * 2 * k, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, batch):
        batch_size = batch.batch.max().item() + 1
        x = batch.fp.view(batch_size, -1)  # [batch_size, input_dim]
        x = x.float()
        x = self.ifm(x)  # Apply IFM
        x = F.relu(self.fc1(x))  # First fully connected layer
        x = self.fc2(x)  # Output layer
        return x

class IFMMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, k=8, sigma=6):
        """
        MLP with IFM layer
        Args:
            input_dim: Original input dimensionality
            hidden_dim: Number of hidden units in the MLP
            output_dim: Output dimensionality
            k: Number of frequency components for IFM
            sigma: Standard deviation for IFM parameters
        """
        super(IFMMLP, self).__init__()
        self.ifm = IFM(input_dim, k, sigma)
        self.fc1 = nn.Linear(input_dim * 2 * k, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = self.ifm(x)  # Apply IFM
        x = F.relu(self.fc1(x))  # First fully connected layer
        x = self.fc2(x)  # Output layer
        return x

class TransformerEmbeddingModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=4, num_heads=8):
        super(TransformerEmbeddingModel, self).__init__()

        self.embed_dim = input_size
        self.hidden_size = hidden_size

        self.embedding = nn.Linear(input_size, hidden_size)
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=self.encoder_layer,
            num_layers=num_layers
        )
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, x):

        x = x.unsqueeze(0)


        x = self.embedding(x)


        x = self.encoder(x)


        x = x.mean(dim=0)


        out = self.out(x)
        # print(out.shape)

        return out

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.hidden_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)



class DNM_Linear_M(nn.Module):    # MDNN
    def __init__(self, input_size, out_size, M=5, device='cpu'):
        super(DNM_Linear_M, self).__init__()

        Synapse_W = torch.rand([out_size, M, input_size]).to(device)#.cuda() # [size_out, M, size_in]
        Synapse_q = torch.rand([out_size, M, input_size]).to(device)#.cuda()
        torch.nn.init.constant_(Synapse_q, 0.1)
        k = torch.rand(1).to(device)
        qs = torch.rand(1).to(device)

        self.params = nn.ParameterDict({'Synapse_W': nn.Parameter(Synapse_W)})
        self.params.update({'Synapse_q': nn.Parameter(Synapse_q)})
        self.params.update({'k': nn.Parameter(k)})
        self.params.update({'qs': nn.Parameter(qs)})
        self.input_size = input_size

    def forward(self, x):
        # Synapse
        out_size, M, _ = self.params['Synapse_W'].shape
        x = torch.unsqueeze(x, 1)
        x = torch.unsqueeze(x, 2)
        x = x.repeat(1, out_size, M, 1)
        x = torch.mul(x, self.params['Synapse_W']) - self.params['Synapse_q']
        x = torch.sigmoid(x)

        # Dendritic
        x = torch.prod(x, 3) #prod 
        # x = torch.tanh(x)

        # Membrane
        x = torch.sum(x, 2)

        # Soma
        x = self.params['k'] * (x - self.params['qs'])

        return x

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)

class MemCell(nn.Module):
    def __init__(self, input_size, hidden_dim,  M, gate=DNM_Linear_M, device='cpu'):
        super(MemCell, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.device = device

        self.update_gate = gate(input_size, hidden_dim, M, device=device)
        self.output_gate = gate(input_size+hidden_dim, hidden_dim, M, device=device)

        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, hidden):

        update = self.update_gate(x)
        update = self.sigmoid(update)
        hidden = update * hidden

        xh = torch.cat([x, hidden], 1)
        output = self.output_gate(xh)
        output = self.tanh(output)
        output = hidden + output

        return output

class DNM_MEM(nn.Module):    # RDNN
    def __init__(self, input_size, hidden_dim, M=2, gate=ADNM_ifm3, device='cpu'):
        super(DNM_MEM, self).__init__()
        self.input_size = input_size
        self.hidden_dim = hidden_dim
        self.device = device
        self.rnn_cell = MemCell(input_size, hidden_dim, M, gate=gate, device=device)

    def forward(self, x):
        """
        :param x: (seq_len, batch, input_size)
        :return:
           output (seq_len, batch, hidden_dim)
           h_n    (1, batch, hidden_dim)
        """
        seq_len, batch, _ = x.shape
        h = torch.zeros(batch, self.hidden_dim).to(self.device)
        output = torch.zeros(seq_len, batch, self.hidden_dim).to(self.device)

        for i in range(seq_len):
            inp = x[i, :, :].to(self.device)
            h = self.rnn_cell(inp, h)
            output[i, :, :] = h

        h_n = output[-1, :, :]
        return output, h_n

class ADNM_ifmT(nn.Module):
    def __init__(self, input_size, out_size, M, device='cpu', k=8, sigma=6):
        super(ADNM_ifmT, self).__init__()
        self.M = M
        self.input_size = input_size
        self.out_size = out_size

        m = torch.rand([out_size, M]).to(device)

        torch.nn.init.uniform_(m, a=-10.0, b=10.0)

        self.params = nn.ParameterDict({
            'm': nn.Parameter(m),
        })
        self.ifm = IFM(input_size, k, sigma)
        self.mlp = nn.Linear(input_size*2*k, out_size * M)
        self.act = nn.Sigmoid()

    def forward(self, x):

        x = self.ifm(x)  # Apply IFM
        x = self.mlp(x)
        x = self.act(x)
        x = x.view(*x.shape[:-1], self.out_size, self.M)
        
        A = torch.tanh(self.params['m'])
        O = torch.sum(x * A, -1) 
        
        return O

    def reset_parameters(self):
        std = 1.0 / math.sqrt(self.input_size)
        for w in self.parameters():
            w.data.uniform_(-std, std)