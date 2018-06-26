from collections import defaultdict

import math
import numpy as np
import torch
from itertools import islice
from tqdm import tqdm

max_sentence_length = 80


def evaluate_multiclass_without_none_ensemble(models, data_test, len_test):
    num_test_instance = 0

    results = defaultdict(dict)
    thres_set_ga = list(map(lambda n: n / 100.0, list(range(20, 71, 1))))
    thres_set_wo = list(map(lambda n: n / 100.0, list(range(20, 86, 1))))
    thres_set_ni = list(map(lambda n: n / 100.0, list(range(0, 61, 1))))

    thres_lists = [thres_set_ga, thres_set_wo, thres_set_ni]
    best_result = init_result_info(results, thres_lists)

    for xss, yss in tqdm(data_test, total=len_test, mininterval=5):
        if yss.size(1) > max_sentence_length:
            continue
        num_test_instance += 1

        scoress = []
        for model in models:
            model = model.eval()
            scoress += [model.cuda()(xss)]
            model.cpu()
        # scoress = [model(xss) for model in models]
        # scores = np.array([model(xss).T for model in models]).mean(axis=0)

        for i in range(yss.size()[0]):
            ys = yss[i]
            predicted = [torch.t(torch.pow(torch.zeros(scores[i].size()).cuda() + math.e, scores[i].data)) for scores in scoress]
            predicted = torch.mean(torch.stack(predicted), 0).cpu()
            # predicted = torch.t(scores[i].cpu())
            # print(torch.pow(torch.zeros(predicted.size()) + math.e, predicted.data))

            for label in range(3):  # case label index
                p_ys = predicted[label]
                add_results_foreach_thres_without_none_ensemble(label, p_ys, results[label], thres_lists[label], ys)

    best_thres, f = calc_best_thres(best_result, results, thres_lists)

    return best_thres, f, num_test_instance


def evaluate_multiclass_without_none(model, data_test, len_test):
    num_test_instance = 0

    results = defaultdict(dict)
    thres_set_ga = list(map(lambda n: n / 100.0, list(range(10, 71, 1))))
    thres_set_wo = list(map(lambda n: n / 100.0, list(range(20, 86, 1))))
    thres_set_ni = list(map(lambda n: n / 100.0, list(range(0, 61, 1))))

    thres_lists = [thres_set_ga, thres_set_wo, thres_set_ni]
    best_result = init_result_info(results, thres_lists)

    model.eval()
    for xss, yss in tqdm(data_test, total=len_test, mininterval=5):
        if yss.size(1) > max_sentence_length:
            continue
        num_test_instance += 1

        scores = model(xss)
        for i in range(yss.size()[0]):
            ys = yss[i]
            predicted = torch.t(scores[i].cpu())
            # print(torch.pow(torch.zeros(predicted.size()) + math.e, predicted.data))

            for label in range(3):  # case label index
                p_ys = predicted[label]
                add_results_foreach_thres_without_none(label, p_ys, results[label], thres_lists[label], ys)

    best_thres, f = calc_best_thres(best_result, results, thres_lists)

    return best_thres, f, num_test_instance


def calc_f_score(result, label_results):
    labels = ["ga", "wo", "ni", "all"]

    for label in range(3):
        p_p = label_results[label]["pp"]
        n_p = label_results[label]["np"]
        p_n = label_results[label]["pn"]
        n_n = label_results[label]["nn"]
        prf = label_results[label]["prf"]

        prf["pp"] = p_p
        ppnp = prf["ppnp"] = p_p + n_p
        pppn = prf["pppn"] = p_p + p_n
        p = prf["p"] = float(p_p) / ppnp if (ppnp > 0) else 0.0
        r = prf["r"] = float(p_p) / pppn if (pppn > 0) else 0.0
        f = prf["f"] = 2 * p * r / (p + r) if (p + r > 0.0) else 0.0

        print(labels[label], '\tp:', round(p * 100, 2), '\tr:', round(r * 100, 2), '\tf1:', round(f * 100, 2), "\t",
              p_p, "\t", ppnp, "\t", pppn)

        result["pp"] += label_results[label]["pp"]
        result["np"] += label_results[label]["np"]
        result["pn"] += label_results[label]["pn"]
        result["nn"] += label_results[label]["nn"]

    p_p = result["pp"]
    n_p = result["np"]
    p_n = result["pn"]
    n_n = result["nn"]
    prf = result["prf"]
    prf["pp"] = p_p
    ppnp = prf["ppnp"] = p_p + n_p
    pppn = prf["pppn"] = p_p + p_n
    p = prf["p"] = float(p_p) / ppnp if (ppnp > 0) else 0.0
    r = prf["r"] = float(p_p) / pppn if (pppn > 0) else 0.0
    f = prf["f"] = 2 * p * r / (p + r) if (p + r > 0.0) else 0.0
    print(labels[3], '\tp:', round(p * 100, 2), '\tr:', round(r * 100, 2), '\tf1:', round(f * 100, 2), "\t",
          p_p, "\t", ppnp, "\t", pppn)
    return f


def calc_best_thres(best_result, results, thres_lists):
    best_thres = [0.0, 0.0, 0.0]
    labels = ["ga", "wo", "ni", "all"]
    print("", flush=True)
    for label in range(3):
        def f_of_label(thres):
            return results[label][thres]["prf"]["f"]

        for thres in thres_lists[label]:
            p_p = results[label][thres]["pp"]
            n_p = results[label][thres]["np"]
            p_n = results[label][thres]["pn"]
            n_n = results[label][thres]["nn"]
            prf = results[label][thres]["prf"]

            prf["pp"] = p_p
            ppnp = prf["ppnp"] = p_p + n_p
            pppn = prf["pppn"] = p_p + p_n
            p = prf["p"] = float(p_p) / ppnp if (ppnp > 0) else 0.0
            r = prf["r"] = float(p_p) / pppn if (pppn > 0) else 0.0
            f = prf["f"] = 2 * p * r / (p + r) if (p + r > 0.0) else 0.0

        best_thres[label] = best_t = max(thres_lists[label], key=f_of_label)
        p = results[label][best_t]["prf"]["p"]
        r = results[label][best_t]["prf"]["r"]
        f = results[label][best_t]["prf"]["f"]
        p_p = results[label][best_t]["prf"]["pp"]
        ppnp = results[label][best_t]["prf"]["ppnp"]
        pppn = results[label][best_t]["prf"]["pppn"]
        print(labels[label], '\tp:', round(p * 100, 2), '\tr:', round(r * 100, 2), '\tf1:', round(f * 100, 2), "\t",
              p_p, "\t", ppnp, "\t", pppn)

        best_result["pp"] += results[label][best_t]["pp"]
        best_result["np"] += results[label][best_t]["np"]
        best_result["pn"] += results[label][best_t]["pn"]
        best_result["nn"] += results[label][best_t]["nn"]
    p_p = best_result["pp"]
    n_p = best_result["np"]
    p_n = best_result["pn"]
    n_n = best_result["nn"]
    prf = best_result["prf"]
    prf["pp"] = p_p
    ppnp = prf["ppnp"] = p_p + n_p
    pppn = prf["pppn"] = p_p + p_n
    p = prf["p"] = float(p_p) / ppnp if (ppnp > 0) else 0.0
    r = prf["r"] = float(p_p) / pppn if (pppn > 0) else 0.0
    f = prf["f"] = 2 * p * r / (p + r) if (p + r > 0.0) else 0.0
    print(labels[3], '\tp:', round(p * 100, 2), '\tr:', round(r * 100, 2), '\tf1:', round(f * 100, 2), "\t",
          p_p, "\t", ppnp, "\t", pppn)
    return best_thres, f


def init_result_info(results, thres_sets):
    for label in range(3):
        for thres in thres_sets[label]:
            results[label][thres] = result_info()
    best_result = result_info()
    return best_result


def add_result(label, p_ys, result, ys):
    # get best in prediction
    sorted_args = p_ys.argsort()[::-1]

    gold_ids = np.where(ys == 1)[0]
    # print("gold ids:", gold_ids)
    best_gold_posi = np.fromiter(map(lambda k: p_ys[k], gold_ids), dtype=np.float).argmax()
    best_gold_id = gold_ids[best_gold_posi]
    k_args = list(islice(filter(lambda id: ys[id] != 1, sorted_args), 1))
    loss = (1.0 - p_ys[best_gold_id] + p_ys[k_args[0]])

    assignment = sorted_args[0]
    if ys[-1] == 1:
        if assignment == len(ys) - 1:
            result[label]["nn"] += 1
        else:
            result[label]["np"] += 1
    elif assignment == len(ys) - 1:
        result[label]["pn"] += 1
    elif ys[assignment] == 1:
        result[label]["pp"] += 1
    else:
        result[label]["np"] += 1
        result[label]["pn"] += 1

    return loss

def add_results_foreach_thres_without_none_ensemble(label, p_ys, result, thres_list, ys):
    # print("p_ys", p_ys)
    # print("ys", ys)

    values, assignments = p_ys.max(0)
    assignment = assignments[0]
    # print("label:", label, "assignment:", assignment, "gold label:", ys[assignment])
    for thres in thres_list:
        prob = values[0] - thres

        if not any(y == label for y in ys):
            if prob < 0:
                result[thres]["nn"] += 1
            else:
                result[thres]["np"] += 1
        elif prob >= 0:
            if ys[assignment] == label:
                result[thres]["pp"] += 1
            else:
                result[thres]["np"] += 1
                result[thres]["pn"] += 1
        else:
            result[thres]["pn"] += 1



def add_results_foreach_thres_without_none(label, p_ys, result, thres_list, ys):
    # print("p_ys", p_ys)
    # print("ys", ys)

    values, assignments = p_ys.max(0)
    assignment = assignments.data[0]
    # print("label:", label, "assignment:", assignment, "gold label:", ys[assignment])
    for thres in thres_list:
        prob = math.pow(math.e, values.data[0]) - thres

        if not any(y == label for y in ys):
            if prob < 0:
                result[thres]["nn"] += 1
            else:
                result[thres]["np"] += 1
        elif prob >= 0:
            if ys[assignment] == label:
                result[thres]["pp"] += 1
            else:
                result[thres]["np"] += 1
                result[thres]["pn"] += 1
        else:
            result[thres]["pn"] += 1


def add_results_foreach_thres(label, p_ys, results, thres_lists, ys):
    for thres in thres_lists[label]:
        ys_predict = np.array(p_ys)
        ys_predict[-1] += thres
        # get best in prediction
        assignment = ys_predict.argmax()

        if ys[-1] == 1:
            if assignment == len(ys) - 1:
                results[label][thres]["nn"] += 1
            else:
                results[label][thres]["np"] += 1
        elif assignment == len(ys) - 1:
            results[label][thres]["pn"] += 1
        elif ys[assignment] == 1:
            results[label][thres]["pp"] += 1
        else:
            results[label][thres]["np"] += 1
            results[label][thres]["pn"] += 1


def result_matrix():
    pps = [0, 0, 0, 0]
    nps = [0, 0, 0, 0]  # false positive
    pns = [0, 0, 0, 0]  # false negative
    nns = [0, 0, 0, 0]
    return {"pps": pps, "nps": nps, "pns": pns, "nns": nns,
            "prf": {"ga": {}, "wo": {}, "ni": {}, "all": {}}}


def best_result_matrix():
    pps = [0, 0, 0, 0]
    nps = [0, 0, 0, 0]  # false positive
    pns = [0, 0, 0, 0]  # false negative
    nns = [0, 0, 0, 0]
    return {"pps": pps, "nps": nps, "pns": pns, "nns": nns,
            "prf": {"ga": {}, "wo": {}, "ni": {}, "all": {}}}


def result_info():
    return {"pp": 0, "np": 0, "pn": 0, "nn": 0,
            "prf": {"pp": 0, "ppnp": 0, "pppn": 0}}
