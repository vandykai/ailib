import torch
import torch.nn as nn
import torch.nn.functional as F

def to_scalar(var):
    return var.view(-1).detach().tolist()[0]

def argmax(vec):
    _, idx = torch.max(vec, 1)
    return to_scalar(idx)

def argmax_batch(vecs):
    _, idx = torch.max(vecs, 1)
    return idx

def log_sum_exp_batch(vecs, dim=1):
    maxi = torch.max(vecs, dim)[0]
    maxi_bc = maxi.unsqueeze(dim)
    recti_ = torch.log(torch.sum(torch.exp(vecs - maxi_bc), dim))
    return maxi + recti_

class CRF(nn.Module):

    def __init__(self, start_tag_id, stop_tag_id, tagset_size):
        super(CRF,self).__init__()

        self.START_TAG_ID = start_tag_id
        self.STOP_TAG_ID = stop_tag_id
        self.tagset_size = tagset_size
        self.transitions = torch.randn(self.tagset_size, self.tagset_size)
        # self.transitions = torch.zeros(tagset_size, tagset_size)
        self.transitions.detach()[self.START_TAG_ID, :] = -10000
        self.transitions.detach()[:, self.STOP_TAG_ID] = -10000
        self.transitions = nn.Parameter(self.transitions)

    def _viterbi_decode(self, feats):
        backpointers = []
        backscores = []
        scores = []
        init_vvars = torch.full((1, self.tagset_size), -10000., device=feats.device)
        init_vvars[0][self.START_TAG_ID] = 0
        forward_var = init_vvars
        for feat in feats:
            next_tag_var = (
                    forward_var.view(1, -1).expand(self.tagset_size, self.tagset_size)
                    + self.transitions
            )
            _, bptrs_t = torch.max(next_tag_var, dim=1)
            viterbivars_t = next_tag_var[range(len(bptrs_t)), bptrs_t]
            forward_var = viterbivars_t + feat
            backscores.append(forward_var)
            backpointers.append(bptrs_t)

        terminal_var = (
                forward_var
                + self.transitions[self.STOP_TAG_ID]
        )
        terminal_var.detach()[self.STOP_TAG_ID] = -10000.0
        terminal_var.detach()[self.START_TAG_ID] = -10000.0
        best_tag_id = argmax(terminal_var.unsqueeze(0))
        best_path = [best_tag_id]
        for bptrs_t in reversed(backpointers):
            best_tag_id = bptrs_t[best_tag_id]
            best_path.append(best_tag_id.item())
        best_scores = []
        for backscore in backscores:
            softmax = F.softmax(backscore, dim=0)
            _, idx = torch.max(backscore, 0)
            prediction = idx.item()
            best_scores.append(softmax[prediction].item())
        start = best_path.pop()
        assert start == self.START_TAG_ID
        best_path.reverse()
        #backscores = [item.detach().cpu().numpy().tolist() for item in backscores]
        return best_scores, best_path

    def _forward_alg(self, feats, lengths):
        init_alphas = torch.full((self.tagset_size, ), -10000.0, device=feats.device)
        init_alphas[self.START_TAG_ID] = 0.0

        forward_var = []
        forward_var.append(init_alphas[None, :].repeat(feats.shape[0], 1))
        transitions = self.transitions.view(
            1, self.transitions.shape[0], self.transitions.shape[1]
        ).repeat(feats.shape[0], 1, 1)
        for i in range(feats.shape[1]):
            emit_score = feats[:, i, :]
            tag_var = (
                emit_score[:, :, None].repeat(1, 1, transitions.shape[2])
                + transitions
                + forward_var[-1][:, :, None]
                .repeat(1, 1, transitions.shape[2])
                .transpose(2, 1)
            )
            forward_var.append(log_sum_exp_batch(tag_var, dim=2))
        forward_var = torch.stack(forward_var).transpose(0,1)
        forward_var = forward_var[range(forward_var.shape[0]), lengths, :]
        terminal_var = forward_var + self.transitions[self.STOP_TAG_ID][None, :].repeat(forward_var.shape[0], 1)
        alpha = log_sum_exp_batch(terminal_var, 1)
        return alpha

    def _score_sentence(self, feats, lengths, tags):
        start = torch.full([tags.shape[0], 1], self.START_TAG_ID, dtype=torch.long, device=tags.device)
        stop = torch.full([tags.shape[0], 1], self.STOP_TAG_ID, dtype=torch.long, device=tags.device)
        pad_start_tags = torch.cat([start, tags], 1)
        pad_stop_tags = torch.cat([tags, stop], 1)
        for i in range(len(lengths)):
            pad_stop_tags[i, lengths[i] :] = self.STOP_TAG_ID
        score = torch.full([feats.shape[0]], 0., dtype=torch.float, device=feats.device)
        for i in range(feats.shape[0]):
            r = torch.tensor(range(lengths[i]), dtype=torch.long, device=feats.device)
            score[i] = torch.sum(
                self.transitions[
                    pad_stop_tags[i, : lengths[i] + 1], pad_start_tags[i, : lengths[i] + 1]
                ]
            ) + torch.sum(feats[i, r, tags[i, : lengths[i]]])
        return score

    def obtain_labels(self, features, input_lens):
        tags = []
        tags_confidences = []
        for feats, length in zip(features, input_lens):
            confidence, tag_seq = self._viterbi_decode(feats[:length])
            tags.append(tag_seq)
            tags_confidences.append(confidence)
        return tags, tags_confidences

    def calculate_loss(self, features, lengths, tags):
        forward_score = self._forward_alg(features, lengths)
        gold_score = self._score_sentence(features, lengths, tags)
        score = forward_score - gold_score
        return score.mean()


