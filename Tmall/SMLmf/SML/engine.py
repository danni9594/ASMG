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

        cur_batch_loader = BatchLoader(cur_set, train_config['base_bs'])

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

    def transfer_train_an_epoch(self, epoch_id, next_set, train_config):

        train_start_time = time.time()

        if train_config['shuffle']:
            next_set = next_set.sample(frac=1)

        next_batch_loader = BatchLoader(next_set, train_config['transfer_bs'])

        transfer_loss_next_sum = 0

        for i in range(1, next_batch_loader.num_batches + 1):

            next_batch = next_batch_loader.get_batch(batch_id=i)

            transfer_loss_next = self.model.train_transfer(self.sess, next_batch)  # sess.run

            if (i - 1) % 100 == 0:
                print('[Epoch {} Batch {}] transfer_loss_next {:.4f}, time elapsed {}'.format(epoch_id,
                                                                                              i,
                                                                                              transfer_loss_next,
                                                                                              time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time))))

                # test the performance of transferred model (can comment out if not needed)
                next_auc, next_logloss = self.test(next_set, train_config)
                print('next_auc {:.4f}, next_logloss {:.4f}'.format(
                    next_auc,
                    next_logloss))
                print('time elapsed {}'.format(time.strftime('%H:%M:%S', time.gmtime(time.time() - train_start_time))))

                print('')

            transfer_loss_next_sum += transfer_loss_next

        # epoch done, compute average loss
        transfer_loss_next_avg = transfer_loss_next_sum / next_batch_loader.num_batches

        return transfer_loss_next_avg

    def test(self, test_set, train_config):

        test_batch_loader = BatchLoader(test_set, train_config['base_bs'])

        scores, losses, labels = [], [], []
        for i in range(1, test_batch_loader.num_batches + 1):
            test_batch = test_batch_loader.get_batch(batch_id=i)
            batch_scores, batch_losses = self.model.inference(self.sess, test_batch)  # sees.run
            scores.extend(batch_scores.tolist())
            losses.extend(batch_losses.tolist())
            labels.extend(test_batch[2])

        test_auc = cal_roc_auc(scores, labels)
        test_logloss = sum(losses) / len(losses)

        return test_auc, test_logloss
