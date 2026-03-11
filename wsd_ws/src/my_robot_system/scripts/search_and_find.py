#!/usr/bin/env python3
import rospy
from active_searcher import ActiveSearcher

if __name__ == '__main__':
    rospy.init_node('active_search_and_find')
    target = rospy.get_param('~target', 'tool_box')
    searcher = ActiveSearcher(target_tag=target)
    searcher.run()
