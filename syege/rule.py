import json
import numpy as np
from collections import defaultdict


def vector2dict(x, feature_names):
    return {k: v for k, v in zip(feature_names, x)}


def record2str(x, feature_names, numeric_columns):
    xd = vector2dict(x, feature_names)
    s = '{ '
    for att, val in xd.items():
        if att not in numeric_columns and val == 0.0:
            continue
        if att in numeric_columns:
            s += '%s = %s, ' % (att, val)
        else:
            att_split = att.split('=')
            s += '%s = %s, ' % (att_split[0], att_split[1])

    s = s[:-2] + ' }'
    return s


class Condition(object):

    def __init__(self, att, op, thr, is_continuous=True):
        self.att = att
        self.op = op
        self.thr = thr
        self.is_continuous = is_continuous

    def __str__(self):
        if self.is_continuous:
            return '%s %s %.2f' % (self.att, self.op, self.thr)
        else:
            att_split = self.att.split('=')
            sign = '=' if self.op == '>' else '!='
            return '%s %s %s' % (att_split[0], sign, att_split[1])

    def __eq__(self, other):
        return self.att == other.att and self.op == other.op and self.thr == other.thr

    def __hash__(self):
        return hash(str(self))


class Rule(object):

    def __init__(self, premises, cons, class_name):
        self.premises = premises
        self.cons = cons
        self.class_name = class_name

    def _pstr(self):
        return '{ %s }' % (', '.join([str(p) for p in self.premises]))

    def _cstr(self):
        if not isinstance(self.class_name, list):
            return '{ %s: %s }' % (self.class_name, self.cons)
        else:
            return '{ %s }' % self.cons

    def __str__(self):
        return '%s --> %s' % (self._pstr(), self._cstr())

    def __eq__(self, other):
        return self.premises == other.premises and self.cons == other.cons

    def __len__(self):
        return len(self.premises)

    def __hash__(self):
        return hash(str(self))

    def is_covered(self, x, feature_names):
        xd = vector2dict(x, feature_names)
        for p in self.premises:
            if p.op == '<=' and xd[p.att] > p.thr:
                return False
            elif p.op == '>' and xd[p.att] <= p.thr:
                return False
        return True


def json2cond(obj):
    return Condition(obj['att'], obj['op'], obj['thr'], obj['is_continuous'])


def json2rule(obj):
    premises = [json2cond(p) for p in obj['premise']]
    cons = obj['cons']
    class_name = obj['class_name']
    return Rule(premises, cons, class_name)


class ConditionEncoder(json.JSONEncoder):
    """ Special json encoder for Condition types """
    def default(self, obj):
        if isinstance(obj, Condition):
            json_obj = {
                'att': obj.att,
                'op': obj.op,
                'thr': obj.thr,
                'is_continuous': obj.is_continuous,
            }
            return json_obj
        return json.JSONEncoder.default(self, obj)


class RuleEncoder(json.JSONEncoder):
    """ Special json encoder for Rule types """
    def default(self, obj):
        if isinstance(obj, Rule):
            ce = ConditionEncoder()
            json_obj = {
                'premise': [ce.default(p) for p in obj.premises],
                'cons': obj.cons,
                'class_name': obj.class_name
            }
            return json_obj
        return json.JSONEncoder.default(self, obj)


def get_rule(x, dt, feature_names, class_name, class_values, numeric_columns):

    x = x.reshape(1, -1)
    feature = dt.tree_.feature
    threshold = dt.tree_.threshold

    leave_id = dt.apply(x)
    node_index = dt.decision_path(x).indices

    premises = list()
    for node_id in node_index:
        if leave_id[0] == node_id:
            break
        else:
            op = '<=' if x[0][feature[node_id]] <= threshold[node_id] else '>'
            att = feature_names[feature[node_id]]
            thr = threshold[node_id]
            iscont = att in numeric_columns
            premises.append(Condition(att, op, thr, iscont))

    dt_outcome = dt.predict(x)[0]
    cons = class_values[int(dt_outcome)]
    premises = compact_premises(premises)
    return Rule(premises, cons, class_name)


def get_rules(dt, feature_names, class_name, class_values, numeric_columns):

    n_nodes = dt.tree_.node_count
    feature = dt.tree_.feature
    threshold = dt.tree_.threshold
    children_left = dt.tree_.children_left
    children_right = dt.tree_.children_right
    value = dt.tree_.value

    is_leaves = np.zeros(shape=n_nodes, dtype=bool)
    stack = [(0, -1)]  # seed is the root node id and its parent depth
    reverse_dt_dict = dict()
    left_right = dict()
    while len(stack) > 0:
        node_id, parent_depth = stack.pop()

        # If we have a test node
        if children_left[node_id] != children_right[node_id]:
            stack.append((children_left[node_id], parent_depth + 1))
            stack.append((children_right[node_id], parent_depth + 1))
            reverse_dt_dict[children_left[node_id]] = node_id
            left_right[(node_id, children_left[node_id])] = 'l'
            reverse_dt_dict[children_right[node_id]] = node_id
            left_right[(node_id, children_right[node_id])] = 'r'
        else:
            is_leaves[node_id] = True

    node_index_list = list()
    for node_id in range(n_nodes):
        if is_leaves[node_id]:
            node_index = [node_id]
            parent_node = reverse_dt_dict.get(node_id, None)
            while parent_node:
                node_index.insert(0, parent_node)
                parent_node = reverse_dt_dict.get(parent_node, None)
            if node_index[0] != 0:
                node_index.insert(0, 0)
            node_index_list.append(node_index)

    if len(value) > 1:
        value = np.argmax(value.reshape(len(value), 2), axis=1)

        rules = list()
        for node_index in node_index_list:

            premises = list()
            for i in range(len(node_index) - 1):
                node_id = node_index[i]
                child_id = node_index[i+1]

                op = '<=' if left_right[(node_id, child_id)] == 'l' else '>'
                att = feature_names[feature[node_id]]
                thr = threshold[node_id]
                iscont = att in numeric_columns
                premises.append(Condition(att, op, thr, iscont))

            cons = class_values[int(value[node_index[-1]])]
            premises = compact_premises(premises)
            rules.append(Rule(premises, cons, class_name))

    else:
        x = np.zeros(len(feature_names)).reshape(1, -1)
        dt_outcome = dt.predict(x)[0]
        cons = class_values[int(dt_outcome)]
        rules = [Rule([], cons, class_name)]
    return rules


def compact_premises(plist):
    att_list = defaultdict(list)
    for p in plist:
        att_list[p.att].append(p)

    compact_plist = list()
    for att, alist in att_list.items():
        if len(alist) > 1:
            min_thr = None
            max_thr = None
            for av in alist:
                if av.op == '<=':
                    max_thr = min(av.thr, max_thr) if max_thr else av.thr
                elif av.op == '>':
                    min_thr = max(av.thr, min_thr) if min_thr else av.thr

            if max_thr:
                compact_plist.append(Condition(att, '<=', max_thr))

            if min_thr:
                compact_plist.append(Condition(att, '>', min_thr))
        else:
            compact_plist.append(alist[0])
    return compact_plist
