import time
from utils import *


class Engine(object):
    """
    Training epoch and test
    """

    def __init__(self, sess, model):

        self.sess = sess
        self.model = model

    def meta_train_an_epoch(self, epoch_id, next_set_ls, train_config):

        train_start_time = time.time()

        num_batches = -int(-len(next_set_ls[-1]) / train_config['meta_bs'])

        next_batch_loader_ls = []
        for period in range(train_config['seq_length']):
            next_set = next_set_ls[period]
            if train_config['shuffle']:
                next_set = next_set.sample(frac=1)
            next_batch_loader_ls.append(BatchLoader2(next_set, num_batches))

        for next_batch_loader in next_batch_loader_ls:
            assert next_batch_loader.num_batches == num_batches

        meta_loss_next_sum = 0

        for i in range(1, num_batches + 1):

            next_batch_ls = []
            for next_batch_loader in next_batch_loader_ls:
                next_batch_ls.append(next_batch_loader.get_batch(batch_id=i))

            meta_loss_next = self.model.train_meta(self.sess, next_batch_ls)  # sess.run

            if (i - 1) % 100 == 0:
                print('[Epoch {} Batch {}] meta_loss_next {:.4f}, time elapsed {}'.format(epoch_id,
                                                                                          i,
                                                                                          meta_loss_next,
                                                                                          time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time))))

                # test the performance of output serving model at every period (very slow, can comment out if not needed)
                next_auc_ls, next_logloss_ls = self.test_all(next_set_ls, train_config)
                for period in range(train_config['seq_length']):
                    print('period {}: next_auc {:.4f}, next_logloss {:.4f}'.format(
                        period,
                        next_auc_ls[period],
                        next_logloss_ls[period]))
                print('time elapsed {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time))))

                print('')

            meta_loss_next_sum += meta_loss_next

        # epoch done, compute average loss
        meta_loss_next_avg = meta_loss_next_sum / num_batches

        return meta_loss_next_avg

    def test_last(self, test_set, train_config):

        test_batch_loader = BatchLoader(test_set, train_config['meta_bs'])

        scores, losses, labels = [], [], []
        for i in range(1, test_batch_loader.num_batches + 1):
            test_batch = test_batch_loader.get_batch(batch_id=i)
            batch_scores, batch_losses = self.model.inference(self.sess, test_batch, train_config['seq_length'] - 1)  # sees.run
            scores.extend(batch_scores.tolist())
            losses.extend(batch_losses.tolist())
            labels.extend(test_batch[4])

        test_auc = cal_roc_auc(scores, labels)
        test_logloss = sum(losses) / len(losses)

        return test_auc, test_logloss

    def test_all(self, test_set_ls, train_config):

        test_auc_ls = []
        test_logloss_ls = []

        for period in range(train_config['seq_length']):

            test_set = test_set_ls[period]

            test_batch_loader = BatchLoader(test_set, train_config['meta_bs'])

            scores, losses, labels = [], [], []
            for i in range(1, test_batch_loader.num_batches + 1):
                test_batch = test_batch_loader.get_batch(batch_id=i)
                batch_scores, batch_losses = self.model.inference(self.sess, test_batch, period)  # sees.run
                scores.extend(batch_scores.tolist())
                losses.extend(batch_losses.tolist())
                labels.extend(test_batch[4])

            test_auc = cal_roc_auc(scores, labels)
            test_logloss = sum(losses) / len(losses)

            test_auc_ls.append(test_auc)
            test_logloss_ls.append(test_logloss)

        return test_auc_ls, test_logloss_ls
