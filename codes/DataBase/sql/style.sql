select t.* from
(
select t.*, rank() over(order by mc desc)lead_stock from
(
select t.*, rank() over(partition by l1_name, l2_name, l3_name order by mc desc) lead_ind from
(
select tl.trade_date, tl.stock_code, tl.price, tl.amount, tsb.name, ti.l1_name, ti.l2_name, ti.l3_name, ts.mc, ts.rank_mc, ts.pb, ts.rank_pb
from label.tdailylabel tl
left join tsdata.ttsstockbasic tsb
on tl.stock_code = tsb.stock_code
left join indsw.tindsw ti
on tl.stock_code = ti.stock_code
left join style.tdailystyle ts
on tl.trade_date = ts.trade_date
and tl.stock_code = ts.stock_code
where tl.trade_date = '20231113'
and tl.amount >= 50000 and tl.price >= 5
and ts.rank_pb >= 0.2 
and ti.l3_name in (
'大众出版', '影视动漫制作', '文字媒体', '视频媒体', '游戏Ⅲ', 
'其他养殖', '生猪养殖', '肉鸡养殖', '农业综合Ⅲ', '其他农产品加工', '果蔬加工', '粮油加工', '水产养殖', '海洋捕捞', '宠物食品', 
'中药Ⅲ', '医院',
'多业态零售', '百货', '超市', '旅游零售Ⅲ', 
'个护小家电', '厨房小家电', '清洁小家电', 
'电动乘用车', '综合乘用车', '其他运输设备', '摩托车', 
'品牌消费电子',
'体育Ⅲ', '人工景区', '旅游综合', '自然景区', '酒店', '餐饮', 
'家纺', '运动服装', '非运动服装', '鞋帽及其他', '其他饰品', '钟表珠宝', 
'洗护用品', '生活用纸', '品牌化妆品', '医美服务', '医美耗材', 
'烘焙食品', '熟食', '零食', '白酒Ⅲ', '调味发酵品Ⅲ', '其他酒类', '啤酒', '保健品', '肉制品', '预加工食品', '乳品', '软饮料')
) t
) t
where t.lead_ind <= 3
) t
where t.lead_stock <= 50
order by l1_name, l2_name, l3_name, lead_stock