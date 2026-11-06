from q1_4v4.course_table import course_rows,course_table_html,COLUMNS


def sample(summary,name,success=1.,stable=1.,mastered=0,worlds=100):
    for key,value in {'成功率':success,'稳定率':stable,'红队成功率':success,'蓝队成功率':success,
                      '红队稳定率':stable,'蓝队稳定率':stable,'已掌握':mastered,'球场数':worlds}.items():
        summary[f'06_课程/{name}/{key}']=value


def test_live_table_tracks_new_values_and_identifies_regression():
    summary={}
    for name in ('balance','balance_push','weight_shift'):sample(summary,name,mastered=1)
    sample(summary,'step_in_place',success=0.)
    sample(summary,'slow_walk',success=0.)
    rows=course_rows(summary)
    assert list(rows[0])==list(COLUMNS) and len(rows)==28
    assert rows[0]['状态']=='复习·达标'
    assert rows[3]['状态']=='主训' and rows[3]['当前成功率']=='0.00%'
    assert rows[4]['状态']=='预习'
    assert rows[5]['状态']=='未解锁' and rows[5]['当前成功率']=='—'
    # A team can fail even when the overall mean appears to pass.
    summary['06_课程/weight_shift/蓝队成功率']=.6
    summary['06_课程/weight_shift/成功率']=.85
    changed=course_rows(summary)
    assert changed[2]['当前成功率']=='85.00%' and changed[2]['状态']=='复习·退化'
    sample(summary,'step_in_place',success=.9)
    assert course_rows(summary)[3]['状态']=='主训·达标待认证'
    sample(summary,'step_in_place',success=.9,mastered=1)
    assert course_rows(summary)[4]['状态']=='主训'


def test_missing_team_is_waiting_for_evidence_not_regression():
    summary={};sample(summary,'balance',mastered=1)
    summary['06_课程/balance/蓝队成功率']=None
    rows=course_rows(summary)
    assert rows[0]['状态']=='复习·待采样'


def test_serial_mode_does_not_present_old_values_as_current_measurements():
    summary={'01_总览/课程编号':3}
    sample(summary,'balance',mastered=1)
    sample(summary,'step_in_place',success=0.)
    rows=course_rows(summary,parallel=False)
    assert rows[0]['状态']=='已通过·无复测' and rows[0]['当前成功率']=='—'
    assert rows[3]['状态']=='主训' and rows[3]['当前成功率']=='0.00%'
    assert rows[4]['状态']=='待训练'
    assert rows[0]['足球场数']=='0' and rows[0]['占比']=='0.00%'
    assert rows[3]['足球场数']=='100' and rows[3]['占比']=='100.00%'


def test_direct_match_table_labels_skipped_courses_without_false_certifications():
    summary={'01_总览/课程编号':27,'01_总览/直接起始课程编号':27,'04_性能/并行球场总数':20480}
    sample(summary,'match',worlds=20480,success=0.)
    rows=course_rows(summary,parallel=False)
    assert all(r['状态']=='本次跳过' and r['足球场数']=='0' for r in rows[:27])
    assert rows[27]['状态']=='主训' and rows[27]['占比']=='100.00%'
    page=course_table_html(summary,1300,parallel=False)
    assert page.index('27 4v4比赛')<page.index('0 站立')


def test_global_field_allocation_is_not_counted_once_per_robot_or_per_gpu():
    summary={'04_性能/并行球场总数':20480,'04_性能/GPU数量':5}
    allocation={'balance':685,'balance_push':685,'weight_shift':685,
                'step_in_place':16365,'slow_walk':1030,'turn':1030}
    for name,worlds in allocation.items():sample(summary,name,worlds=worlds)
    rows=course_rows(summary)
    assert sum(int(r['足球场数']) for r in rows)==20480
    assert rows[3]['足球场数']=='16365' and rows[3]['占比']=='79.91%'
    assert rows[4]['足球场数']=='1030' and rows[4]['占比']=='5.03%'
    assert rows[27]['足球场数']=='0' and rows[27]['占比']=='0.00%'
    page=course_table_html(summary,1029)
    assert '20,480个' in page and '正在采样：6门课程' in page


def test_html_has_six_columns_and_iteration_and_no_external_runtime():
    from html.parser import HTMLParser
    class Counter(HTMLParser):
        heads=0;rows=0
        def handle_starttag(self,tag,attrs):
            if tag=='th':self.heads+=1
            if tag=='tr':self.rows+=1
    page=course_table_html({},123)
    parsed=Counter();parsed.feed(page)
    assert parsed.heads==6 and parsed.rows==29
    assert '第123轮' in page and '<script' not in page and 'https://' not in page
    import wandb
    media=wandb.Html(page,inject=False,data_is_not_path=True)
    assert media.html==page
