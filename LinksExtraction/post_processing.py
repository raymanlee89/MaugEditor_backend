def is_in_range(loc, range):
    if range["start"] <= loc and loc <= range["end"]:
        return True
    return False
    
def is_overlapping(e1, e2):
    if is_in_range(e1["start"], e2) or is_in_range(e1["end"], e2):
        return True
    if is_in_range(e2["start"], e1) or is_in_range(e2["end"], e1):
        return True
    return False

def get_union(e1, e2): # e1 should be merged entity
    e1["contain"].append(e2["eid"])
    label = "SYMBOL"
    if e1["label"] == "PRIMARY" or e2["label"] == "PRIMARY":
        label = "PRIMARY"
    return {
        "label": label,
        "text": e1["text"] + e2["text"], # this text may have be incorrect, please use start & end
        "start": min(e1["start"], e2["start"]),
        "end": max(e1["end"], e2["end"]),
        "contain": e1["contain"]
    }

def get_overlapping_entity_groups(entities):
    result = []
    entities_list = sorted(list(entities.values()), key=lambda e: e["start"])
    skip_eid = []
    for eid in range(len(entities_list)):
        if eid in skip_eid:
            continue
        e = entities_list[eid]
        group = {
            "label": e["label"],
            "text": e["text"],
            "start": e["start"],
            "end": e["end"],
            "contain": [e["eid"]]
        }
        for eid2 in range(eid+1, len(entities_list)):
            e2 = entities_list[eid2]
            if is_overlapping(group, e2):
                # print("\nMerge:", group["contain"], e2["eid"])
                # print("Range:", group["start"], group["end"], "<=", e2["start"], e2["end"])
                group = get_union(group, e2)
                skip_eid.append(eid2)
        result.append(group)
    return result

def get_group_id(eid, entity_groups):
    for gid in range(len(entity_groups)):
        if eid in entity_groups[gid]["contain"]:
            return gid
    return -1

def post_processing(entities, relations):
    entity_groups = get_overlapping_entity_groups(entities)
    # print("\nentity_groups:", entity_groups)
    initial_links = []
    for group in entity_groups:
        initial_links.append(group["contain"])

    for r in list(relations.values()):
        # do not merge the Corefer-Description
        if r["label"] == "Corefer-Description":
            continue
        new_link = [r["arg0"], r["arg1"]]
        del_lid = []
        for lid in range(len(initial_links)):
            if r["arg0"] in initial_links[lid] or r["arg1"] in initial_links[lid]:
                new_link = list(set(new_link + initial_links[lid]))
                del_lid.append(lid)
        del_lid.reverse()
        for lid in del_lid:
            initial_links.pop(lid)
        initial_links.append(new_link)
    # print("\ninitial_links:", initial_links)
    links_with_loc = []
    for l in initial_links:
        item = set()
        link = []
        for eid in l:
            gid = get_group_id(eid, entity_groups)
            item.add(gid)
        for gid in item:
            link.append({
                "label": entity_groups[gid]["label"],
                "text": entity_groups[gid]["text"],
                "start": entity_groups[gid]["start"],
                "end": entity_groups[gid]["end"]
            })
        link = sorted(link, key=lambda g: g["start"])
        links_with_loc.append(link)
    return links_with_loc
