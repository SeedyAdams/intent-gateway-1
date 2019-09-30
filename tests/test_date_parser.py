###
# Intent Gateway is a natural language processing (NLP) framework to conduct model development and
# model deployment for text classification.
# Copyright (C) 2018-2019  Asurion, LLC
#
# Intent Gateway is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# Intent Gateway is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Intent Gateway.  If not, see <https://www.gnu.org/licenses/>.
###


"""
date parser tests
"""
import pytest
from datetime import datetime
from IntentGateway import duckparser


dates_testset = """
yesterday	
today	
2/20/2017	
2/20/201	User missed last digit of year
2/20/17	2 digit year
2/20	no year
January	month only (assume middle of month?)
last week	assume middle of week?
last thanksgiving	holiday that changes every year
presidents day	holiday that changes every year
tonight	
this morning	
sunday evening	
Thursday 3-30-17	
Thursday	
Feb-17	month abbreviation with hyphen
valentines day	holiday on same day every year
2 days back	prod user actually said this
2 days ago	
"""


def generate_dates():
	for line in dates_testset.split("\n"):
		yield line

@pytest.fixture(scope='module')
def dateparser():
	print "loading date parser"
	duckparser.init_duck()

@pytest.fixture(scope='module')
def qa_date_cases():
	print "loading test cases filed by QA"
	qa_db = {}
	return qa_db


class TestDates(object):
	
	def test_date_yesterday(self):
		date_today = datetime.today()
		date_extracted = duckparser.extract_datetime("yesterday")
		_datetime_parse = datetime.strptime(date_extracted["yesterday"],"%Y-%m-%dT%H:%M:%S.%fZ")
		assert _datetime_parse.year == date_today.year
		if _datetime_parse.month == date_today.month and date_today.day != 1:
			assert _datetime_parse.day == date_today.day - 1
		with pytest.raises(ValueError) as excinfo:
			_datetime_parse = datetime.strptime(date_extracted["yesterday"][:5],"%Y-%m-%dT%H:%M:%S.%fZ")
		assert "does not match format" in str(excinfo.value)

	def test_date_today(self):
		date_today = datetime.today()
		date_extracted = duckparser.extract_datetime("today")
		_datetime_parse = datetime.strptime(date_extracted["today"],"%Y-%m-%dT%H:%M:%S.%fZ")
		assert _datetime_parse.year == date_today.year
		assert _datetime_parse.month == date_today.month
		assert _datetime_parse.day == date_today.day

	def test_mmddyy(self):
		test_mmddyy_dates = ["2/20/2017", "2/20/17"]
		for d in test_mmddyy_dates:
			date_extracted = duckparser.extract_datetime(d)
			_datetime_parse = datetime.strptime(date_extracted[d],"%Y-%m-%dT%H:%M:%S.%fZ")
			assert _datetime_parse.year == 2017
			assert _datetime_parse.month == 2
			assert _datetime_parse.day == 20

	def test_simple_dates(self, dateparser, qa_date_cases):
		assert len(qa_date_cases) == 0
		
	def test_qa_discovered_bugs(self, dateparser):
		for date in generate_dates():
			# todo: get data from vishal
			assert date == date

	def test_luis_bugs(self, dateparser):
		for date in generate_dates():
			# todo: get data from Mike
			assert date == date
