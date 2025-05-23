def lr_schedule(optimizer,batch_num,step,alpha):
    if batch_num != 0 and batch_num % step == 0:
        for group in optimizer.param_groups:
            group['lr'] *= alpha
        print(f"learning rate = {optimizer.param_groups[0]['lr']}")    
            