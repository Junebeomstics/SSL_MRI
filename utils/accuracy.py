def accuracy(output, target, topk=(1,)):
    """
    Given predicted and ground truth labels, 
    calculate top-k accuracies.
    
    Args:
      output: predicted labels
      target: ground truth labels
      topk: a tuple with the top-n accuracies to calculate
      
    Returns:
      top-k accuracies
    """
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def save_checkpoint(self, epoch, best_prec, is_best):
        state = {
                'epoch': epoch,
                'best_prec': best_prec,
                'state_dict': self.model.state_dict(),
                'optim' : self.optim.state_dict(),
                }
        utils.save_checkpoint(state, is_best, dirpath=self.dirpath, filename='model_checkpoint.pkl')
        if(is_best):
            path_save = os.path.join(self.dirpath, 'weight_best.pkl')
            torch.save({'state_dict': self.model.state_dict()}, path_save) 