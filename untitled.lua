require 'torch'

     function interval_overlap(gts, dets)
  local num_gt = #gts
  local num_det = #dets
  local ov = torch.Tensor(num_gt, num_det)
  for i=1,num_gt do
    for j=1,num_det do
      ov[i][j] = interval_overlap_single(gts[i], dets[j])
    end
  end
  return ov
end

function interval_overlap_single(gt, dt)
  local i1 = gt
  local i2 = dt
  -- union
  local bu = {math.min(i1[1], i2[1]), math.max(i1[2], i2[2])}
  local ua = bu[2] - bu[1]
  -- overlap
  local ov = 0
  local bi = {math.max(i1[1], i2[1]), math.min(i1[2], i2[2])}
  local iw = bi[2] - bi[1]
  if iw > 0 then
    ov = iw / ua
  end
  return ov
end

function round(num, idp)
  local mult = 10^(idp or 0)
  return math.floor(num * mult + 0.5) / mult
end

        local tpconf={0.8,0.9}
        local fpconf={0.85}
        local fnconf={1,1,1}
        
        
        
   local randomize_ap=true
    local num_tp = #tpconf
    local num_fp = #fpconf
    local num_fn = #fnconf
local conf = torch.Tensor(2, num_tp+num_fp+num_fn):zero()
    for i=1,num_tp do
        conf[1][i] = round(tpconf[i],2)
        conf[2][i] = 1
    end
    for i=1,num_fp do
        conf[1][i+num_tp] = round(fpconf[i],2)
        conf[2][i+num_tp] = 2
    end
    for i=1,num_fn do
        conf[1][i+num_tp+num_fp] = fnconf[i]
        conf[2][i+num_tp+num_fp] = 3
    end
    if num_tp+num_fp+num_fn == 0 then
        return 0
    end
    local _,sorted_idxs=torch.sort(conf[1], true)
    sorted_conf = torch.Tensor(2, num_tp+num_fp+num_fn)
    for i=1,sorted_idxs:size(1) do
        sorted_conf[1][i] = conf[1][sorted_idxs[i]]
        sorted_conf[2][i] = conf[2][sorted_idxs[i]]
    end
    local sorted_conf_counts = {}
    local sorted_conf_vals = {}
    local prev_conf_val = 0
    for i=1,sorted_idxs:size(1) do
        local conf_val = sorted_conf[1][i]
        local conf_type = sorted_conf[2][i]
        local new_val = false
        if conf_val ~= prev_conf_val then
            new_val = true
        end
        if new_val then
            table.insert(sorted_conf_vals, conf_val)
            sorted_conf_counts[conf_val] = {0, 0,0}
            prev_conf_val = conf_val
        end
        sorted_conf_counts[conf_val][conf_type] = sorted_conf_counts[conf_val][conf_type] + 1
    end

    local conf_iter = 1
    for i=1,#sorted_conf_vals do
        local conf_val = sorted_conf_vals[i]
        local conf_counts = sorted_conf_counts[conf_val]
        local num_tp_counts = conf_counts[1]
        local num_fp_counts = conf_counts[2]
        local num_fn_counts = conf_counts[3]--conf type 1:tp 2:fp 3:fn
        local total_conf_counts = num_tp_counts + num_fp_counts +num_fn_counts

        sorted_conf[{1, {conf_iter, conf_iter+total_conf_counts-1}}]:fill(conf_val)

        if num_tp_counts > 0 then
            sorted_conf[{2, {conf_iter, conf_iter+num_tp_counts-1}}]:fill(1)
        end
        if num_fp_counts > 0 then
            sorted_conf[{2, {conf_iter+num_tp_counts, conf_iter+num_tp_counts+num_fp_counts-1}}]:fill(2)
        end
         if num_fn_counts > 0 then
           ---still vaguelly
            sorted_conf[{2, {conf_iter+num_tp_counts+num_fp_counts, conf_iter+total_conf_counts-1}}]:fill(3)
        end

        if randomize_ap then
            local shuffle = torch.randperm(total_conf_counts):type('torch.LongTensor')
            sorted_conf[{2, {conf_iter, conf_iter+total_conf_counts-1}}] = sorted_conf[{2, {conf_iter, conf_iter+total_conf_counts-1}}]:index(1, shuffle)
        end
        conf_iter = conf_iter + total_conf_counts
    end
    tp = torch.cumsum(sorted_conf[2]:eq(1):double())
    fp = torch.cumsum(sorted_conf[2]:eq(2):double())
    -----still vaguelly
    fn = torch.cumsum(sorted_conf[2]:eq(3):double())
    tmp = sorted_conf[2]:eq(1):double()
  --  rec = torch.div(tp, total_num_gt)
    prec = torch.cdiv(2*tp, tp+fp+fn)
    ap = 0
    for i=1,prec:size(1) do
        if tmp[i] == 1 then
            ap = ap + prec[i]
        end
    end
    ap = ap / 3
    print(ap)

    

