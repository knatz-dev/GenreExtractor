"""Confusion matrix helper.

Replaces the abandoned ``pandas_ml.confusion_matrix`` package, which cannot be
imported on Python 3.12+ (it depends on the removed ``distutils`` module).

Only the small API surface used by this project is provided:
construction from ``(y_true, y_pred)``, ``stats()``, ``to_dataframe()`` and
``to_array()``.
"""

import collections

import numpy as np
import pandas as pd


class ConfusionMatrix(object):
    """Confusion matrix for binary and multi-class predictions.

    Attributes
    ----------
    classes : list
        Sorted union of the true and predicted labels.
    """

    TRUE_NAME = 'Actual'
    PRED_NAME = 'Predicted'

    def __init__(self, y_true, y_pred, labels=None,
                 true_name=TRUE_NAME, pred_name=PRED_NAME):
        self.true_name = true_name
        self.pred_name = pred_name

        self._y_true = pd.Series(list(y_true), name=self.true_name)
        self._y_pred = pd.Series(list(y_pred), name=self.pred_name)

        if len(self._y_true) != len(self._y_pred):
            raise ValueError("y_true must have same size as y_pred - {} != {}".format(
                len(self._y_true), len(self._y_pred)))

        self.classes = sorted(set(self._y_true) | set(self._y_pred), key=str)

        df = pd.crosstab(self._y_true, self._y_pred)
        df = df.reindex(index=self.classes, columns=self.classes).fillna(0)
        df = df.astype(np.int64)
        df.index.name = self.true_name
        df.columns.name = self.pred_name
        self._df = df

    def to_dataframe(self, normalized=False, calc_sum=False):
        """Return the confusion matrix as a :class:`pandas.DataFrame`."""
        if normalized:
            values = self._df.values.astype(float)
            row_sums = values.sum(axis=1)[:, np.newaxis]
            values = np.divide(values, row_sums, out=np.zeros_like(values),
                               where=row_sums != 0)
            df = pd.DataFrame(values, index=self._df.index.copy(),
                              columns=self._df.columns.copy())
        else:
            df = self._df.copy()

        if calc_sum:
            df['__all__'] = df.sum(axis=1)
            df = pd.concat([df, pd.DataFrame(df.sum(axis=0),
                                             columns=['__all__']).T])

        df.index.name = self.true_name
        df.columns.name = self.pred_name
        return df

    def to_array(self, normalized=False, sum=False):
        """Return the confusion matrix as a :class:`numpy.ndarray`."""
        return self.to_dataframe(normalized, sum).values

    def _binary_counts(self):
        if len(self.classes) != 2:
            raise ValueError(
                "binary statistics require exactly two classes, got {}".format(
                    self.classes))

        pos_class = self.classes[1]
        neg_class = self.classes[0]

        tp = int(self._df.loc[pos_class, pos_class])
        tn = int(self._df.loc[neg_class, neg_class])
        fp = int(self._df.loc[neg_class, pos_class])
        fn = int(self._df.loc[pos_class, neg_class])
        return tp, tn, fp, fn

    @staticmethod
    def _divide(numerator, denominator):
        if not denominator:
            return float('nan')
        return float(numerator) / float(denominator)

    def stats(self, lst_stats=None):
        """Return an ordered dictionary of binary classification statistics."""
        tp, tn, fp, fn = self._binary_counts()

        population = tp + tn + fp + fn
        condition_positive = tp + fn
        condition_negative = tn + fp
        positive_test = tp + fp
        negative_test = tn + fn

        tpr = self._divide(tp, condition_positive)
        tnr = self._divide(tn, condition_negative)
        ppv = self._divide(tp, positive_test)
        npv = self._divide(tn, negative_test)
        fpr = self._divide(fp, fp + tn)
        fdr = self._divide(fp, fp + tp)
        fnr = self._divide(fn, fn + tp)
        accuracy = self._divide(tp + tn, population)
        f1_score = self._divide(2 * tp, 2 * tp + fp + fn)

        mcc_denominator = np.sqrt((tp + fp) * (tp + fn) *
                                  (tn + fp) * (tn + fn))
        mcc = self._divide((tp * tn) - (fp * fn), mcc_denominator)

        informedness = tpr + tnr - 1
        markedness = ppv + npv - 1
        prevalence = self._divide(condition_positive, population)
        lr_positive = self._divide(tpr, fpr)
        lr_negative = self._divide(fnr, tnr)
        dor = self._divide(lr_positive, lr_negative)
        false_omission_rate = self._divide(fn, fn + tn)

        stats = collections.OrderedDict([
            ('population', population),
            ('P', condition_positive),
            ('N', condition_negative),
            ('PositiveTest', positive_test),
            ('NegativeTest', negative_test),
            ('TP', tp),
            ('TN', tn),
            ('FP', fp),
            ('FN', fn),
            ('TPR', tpr),
            ('TNR', tnr),
            ('PPV', ppv),
            ('NPV', npv),
            ('FPR', fpr),
            ('FDR', fdr),
            ('FNR', fnr),
            ('ACC', accuracy),
            ('F1_score', f1_score),
            ('MCC', mcc),
            ('informedness', informedness),
            ('markedness', markedness),
            ('prevalence', prevalence),
            ('LRP', lr_positive),
            ('LRN', lr_negative),
            ('DOR', dor),
            ('FOR', false_omission_rate),
        ])

        if lst_stats is not None:
            return collections.OrderedDict((key, stats[key]) for key in lst_stats)
        return stats
