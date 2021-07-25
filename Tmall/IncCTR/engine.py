import time
from utils import *


class Engine(object):
    """
    Training epoch and test
    """

    def __init__(self, sess, model):

        self.sess = sess
        self.model = model

    def base_train_an_epoch(self, epoch_id, cur_set, train_config):

        train_start_time = time.time()

        if train_config['shuffle']:
            cur_set = cur_set.sample(frac=1)

        cur_batch_loader = BatchLoaderYsoft(cur_set, train_config['base_bs'])

        base_loss_cur_sum = 0

        for i in range(1, cur_batch_loader.num_batches + 1):

            cur_batch = cur_batch_loader.get_batch(batch_id=i)

            base_loss_cur = self.model.train_base(self.sess, cur_batch)  # sess.run

            if (i - 1) % 100 == 0:
                print('[Epoch {} Batch {}] base_loss_cur {:.4f}, time elapsed {}'.format(epoch_id,
                                                                                         i,
                                                                                         base_loss_cur,
                                                                                         time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time))))

            base_loss_cur_sum += base_loss_cur

        # epoch done, compute average loss
        base_loss_cur_avg = base_loss_cur_sum / cur_batch_loader.num_batches

        return base_loss_cur_avg

    def test(self, test_set, train_config):

        test_batch_loader = BatchLoader(test_set, train_config['base_bs'])

        scores, losses, labels = [], [], []
        for i in range(1, test_batch_loader.num_batches + 1):
            test_batch = test_batch_loader.get_batch(batch_id=i)
            batch_scores, batch_losses = self.model.inference(self.sess, test_batch)  # sees.run
            scores.extend(batch_scores.tolist())
            losses.extend(batch_losses.tolist())
            labels.extend(test_batch[4])

        test_auc = cal_roc_auc(scores, labels)
        test_logloss = sum(losses) / len(losses)

        return test_auc, test_logloss
