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
import numpy as np
from sklearn.neural_network import MLPClassifier
import time
import sys
import pickle


class Classifier(object):
    """
    A custom classifier module for import by nmeta2
    """
    def __init__(self, logger):
        """
        Initialise the classifier
        """
        self.logger = logger
        self.nn_classifier = pickle.load(open('nn_0','rb'))
        self.f = open('neural_net_results.csv', 'w+')
        f.write('before_time_features,before_time_classifier,after_time,result\n')

    def classifier(self, flow):
        """
        A really basic statistical classifier to demonstrate ability
        to differentiate 'bandwidth hog' flows from ones that are
        more interactive so that appropriate classification metadata
        can be passed to QoS for differential treatment.
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
        _max_packets = 10
        #*** Thresholds used in calculations:
        _max_packet_size_threshold = 1200
        _interpacket_ratio_threshold = 0.3

        #*** Dictionary to hold classification results:
        _results = {}

        if flow.packet_count >= _max_packets and not flow.finalised:
            #*** Reached our maximum packet count so do some classification:
            self.logger.debug("Reached max packets count, finalising")
            flow.finalised = 1

            before_time_features = time.time()
            #*** Call functions to get statistics to make decisions on:
            _max_packet_size = flow.max_packet_size()
            _min_packet_size = flow.min_packet_size()
            _avg_packet_size = flow.avg_packet_size()
            _max_interpacket_interval = flow.max_interpacket_interval()
            _min_interpacket_interval = flow.min_interpacket_interval()
            _avg_interpacket_interval = flow.avg_interpacket_arrival()
            _delta_bytes = flow.delta_bytes()
            _duration_flow = flow.duration()
            _byte_velocity = (float)delta_bytes/(float)duration_flow
            _packet_velocity = (float)_max_packets/(float)duration_flow

            features = np_array.array([_duration_flow,_delta_bytes,_avg_interpacket_interval,_min_interpacket_interval,_max_interpacket_interval,_avg_packet_size,_min_packet_size,_max_packet_size, _packet_velocity,_byte_velocity]).astype(float32) 
            
            before_time_classify = time.time()
            result = self.nn_classifier.predict(features.reshape(1,-1))[0]   
            after_time = time.time()
            self.f.write('{0:.3f,{1:.3f,{2:.3f},{3}}\n'.format(before_time_features,before_time_classify,after_time,result)
            

        return _results
