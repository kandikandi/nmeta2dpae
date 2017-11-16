# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
# implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module is part of the nmeta2 suite
.
It defines a custom traffic classifier
.
To create your own custom classifier, copy this example to a new
file in the same directory and update the code as required.
Call it from nmeta by specifying the name of the file (without the
.py) in main_policy.yaml
.
Classifiers are called per packet, so performance is important
.
"""

#*** Required for payload HTTP decode:
import dpkt

class Classifier(object):
    """
    A custom classifier module for import by nmeta2
    """
    def __init__(self, logger):
        """
        Initialise the classifier
        """
        self.logger = logger
	self.f = open('dpi.csv', 'w+')
        self.f.write('before_time_features,before_time_classifier,after_time,result\n')

    def classifier(self, flow):
        """
        A really basic HTTP URI classifier to demonstrate ability
        to differentiate based on a payload characteristic.
        .
        This method is passed a Flow class object that holds the
        current context of the flow
        .
        It returns a dictionary specifying a key/value of QoS treatment to
        take (or not if no classification determination made).
        .
        Only works on TCP.
        """
        #*** Maximum packets to accumulate in a flow before making a
        #***  classification:
        _max_packets = 5

        #*** URI to match:
        _match_uri = '/static/index.html'

        #*** QoS actions to take:
        _qos_action_match = 'constrained_bw'
        _qos_action_no_match = 'default_priority'

        #*** Dictionary to hold classification results:
        _results = {}
        http = ''
	results = ''

        if not flow.finalised:
            #*** Do some classification:
            self.logger.debug("Checking packet")
	    before_time_features = time.time()
            #*** Get the latest packet payload from the flow class:
            payload = flow.payload

            #*** Check if the payload is HTTP:
            if len(payload) > 0:
                try:
		    before_time_classify = time.time()
                    http = dpkt.http.Request(payload)
		    after_time = time.time()
                except:
                    #*** not HTTP so ignore...
                    pass

            if http:
		results = http.uri
                #*** Decide actions based on the URI:
                if http.uri == _match_uri:
                    #*** Matched URI:
                    self.logger.debug("Matched HTTP uri=%s", http.uri)
                    _results['qos_treatment'] = _qos_action_match
                else:
                    #*** Doesn't match URI:
                    self.logger.debug("Did not match HTTP uri=%s", http.uri)
                    _results['qos_treatment'] = _qos_action_no_match

                self.logger.debug("Decided on results %s", _results)

            else:
		results = 'unknown'
                self.logger.debug("Not HTTP so ignoring")

            if flow.packet_count >= _max_packets:
                flow.finalised = 1

            self.f.write('{0},{1},{2},{3},{4:.3f},{5:.3f},{6:.3f},{7}\n'.format(flow.ip_src,flow.tcp_src,flow.ip_dst,flow.tcp_dst,before_time_features,before_time_classify,after_time,results))

        return _results
