# -*- coding: utf-8 -*-

import datetime


def get_timestamp(stamp_format = 'full'):
    """
    Returns a time stamp as a string following the given format:
        'full'        'YYYY-MM-DD HH:MM:SS'
        'full_tight'  'YYYY-MM-DD_HH:MM:SS'
        'date'        'YYYY-MM-DD'
        'date_tight'  'YYYYMMDD
        'time'        'HH:MM:SS'
        'time_tight'  'HHMMSS'
    """

    string_date = str(datetime.datetime.now()).split('.')[0]

    if stamp_format == 'full':
        final_string = string_date
    elif stamp_format == 'full_tight':
        st1 = string_date.split(' ')[0]
        st2 = string_date.split(' ')[1]
        final_string = ''.join(st1.split('-')) + '_' + ''.join(st2.split(':'))
    elif stamp_format == 'date':
        final_string = string_date.split(' ')[0]
    elif stamp_format == 'date_tight':
        st = string_date.split(' ')[0]
        final_string = ''.join(st.split('-'))
    elif stamp_format == 'time':
        final_string = string_date.split(' ')[1]
    elif stamp_format == 'time_tight':
        st = string_date.split(' ')[1]
        final_string = ''.join(st.split(':'))

    return final_string