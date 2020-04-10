import os
import sys
import inspect
from urllib.parse import urlparse
import pandas as pd
import numpy as np
import sklearn
import keras
from sklearn.model_selection import train_test_split
from cli_code.cli_download import Downloader
from collections.abc import MutableMapping
import json
import cloudpickle as pickle
from util import load_callable_from_dict


class PreprocessorError(Exception):
    pass


class MissingDataPreprocessorError(PreprocessorError):
    def __init__(self):
        print(f"data_preprocessor is missing in preprocessor.")


class EncoderMissingIndexError(PreprocessorError):
    def __init__(self, encoder_pars):
        print(f"'{encoder_pars}' is missing the index parameter.")


class EncoderMissingEncoderError(PreprocessorError):
    def __init__(self, encoder_pars):
        print(f"'{encoder_pars}' is missing the encoder parameter.")


class PreprocessorNotFittedError(PreprocessorError):
    def __init__(self):
        print(f"""Preprocessor has not been fitted.""")


class EncoderOutputSizeError(PreprocessorError):
    def __init__(self, output_name, output_size):
        print(
            f"""Encoder output size does not match the number of specified output names {output_name} ({len(output_name)}!={output_size})."""
        )


class PreprocssingOutputDict(dict):
    def __init__(self, data, *args, **kwargs):
        super(PreprocssingOutputDict, self).__init__(*args, **kwargs)
        self.data = data

    def __getitem__(self, key):
        try:
            if isinstance(key, list):
                return (super(PreprocssingOutputDict, self).__getitem__(k) for k in key)
            else:
                return super(PreprocssingOutputDict, self).__getitem__(key)
        except KeyError:
            if isinstance(key, str) or (
                isinstance(key, list) and isinstance(key[0], str)
            ):
                return self.data[key]
            else:
                try:
                    return self.data[:, key]
                except:
                    return self.data[key]

    def __setitem__(self, key, value):
        if isinstance(key, list):
            (
                super(PreprocssingOutputDict, self).__setitem__(k, v)
                for k, v in zip(key, value)
            )
        else:
            super(PreprocssingOutputDict, self).__setitem__(key, value)

    def __str__(self):
        return super(PreprocssingOutputDict, self).__str__()

    def __repr__(self):
        return super(PreprocssingOutputDict, self).__repr__()

    def values(self):
        return_original_object = True
        if isinstance(self.data, pd.DataFrame):
            for key in super(PreprocssingOutputDict, self).keys():
                if key in self.data.columns:
                    try:
                        self.data[key] = self[key]
                    except:
                        return_original_object = False
        elif isinstance(self.data, np.ndarray):
            for key in super(PreprocssingOutputDict, self).keys():
                if isinstance(key, int):
                    try:
                        self.data[:, key] = self[key]
                    except:
                        return_original_object = False
        else:
            for key in super(PreprocssingOutputDict, self).keys():
                if isinstance(key, int):
                    try:
                        self.data[key] = self[key]
                    except:
                        return_original_object = False
        if return_original_object:
            return self.data
        else:
            return super(PreprocssingOutputDict, self).values()


class Preprocessor:
    output_dict_type = PreprocssingOutputDict

    def __init__(self, preprocessor_dict):
        self._preprocessor_specs = []
        self._preprocessors = None
        self._interpret_preprocessor_dict(preprocessor_dict)

    def _interpret_preprocessor_dict(self, preprocessor_dict):
        if isinstance(preprocessor_dict, list):
            for pars in preprocessor_dict:
                preprocessed_data = self._interpret_preprocessor(pars)
        else:
            self._interpret_preprocessor(preprocessor_dict)

    def _interpret_preprocessor(self, pars):
        if len(pars.keys()) == 0:
            return None
        try:
            data_preprocessor = pars.get("data_preprocessor")
        except KeyError:
            raise MissingDataPreprocessorError()
        if isinstance(data_preprocessor, list):
            encoders = []
            for encoder_pars in data_preprocessor:
                encoders.append(self._interpret_encoder(encoder_pars))
            self._preprocessor_specs.append(encoders)
        else:
            self._preprocessor_specs.append(load_callable_from_dict(data_preprocessor))

    def _interpret_encoder(self, encoder_pars):
        try:
            index = encoder_pars.get("index")
        except:
            raise EncoderMissingIndexError(encoder_pars)
        try:
            encoder_str = encoder_pars.get("encoder")
        except:
            raise EncoderMissingEncoderError(encoder_pars)
        output_name = encoder_pars.get("output_name", index)
        encoder, args = load_callable_from_dict(encoder_str)
        return index, encoder, output_name, args

    def fit_transform(self, data):
        self._preprocessors = []
        for preprocessor_spec in self._preprocessor_specs:
            if isinstance(preprocessor_spec, list):
                output = PreprocssingOutputDict(data)
                encoders = []
                for encoder_spec in preprocessor_spec:
                    index, encoder, output_name, args = encoder_spec
                    if isinstance(index, str) or (
                        isinstance(index, list) and isinstance(index[0], str)
                    ):
                        selection = data[index]
                    else:
                        try:
                            selection = data[:, index]
                        except:
                            selection = data[index]
                    if inspect.isclass(encoder):
                        preprocessor_instance = encoder.fit(**args)
                        encoders.append(
                            (index, preprocessor_instance.transform, output_name)
                        )
                        out = preprocessor_instance.transform(selection)
                    else:
                        transform = (
                            (lambda x: encoder(x, **args))
                            if args is not None
                            else lambda x: encoder(x)
                        )
                        encoders.append((index, transform, output_name))
                        out = transform(selection)
                    if (
                        (
                            isinstance(output_name, list)
                            or isinstance(output_name, tuple)
                        )
                        and (isinstance(out, list) or isinstance(out, tuple))
                        and len(output_name) != len(out)
                    ):
                        raise EncoderOutputSizeError(output_name, len(out))
                    output[output_name] = out
                data = output
                self._preprocessors.append(encoders)
            else:
                preprocessor, args = preprocessor_spec
                if inspect.isclass(preprocessor):
                    preprocessor_instance = preprocessor.fit(**args)
                    self._preprocessors.append(preprocessor_instance.transform)
                    data = preprocessor_instance.transform(data)
                else:
                    transform = (
                        (lambda x: preprocessor(x, **args))
                        if args is not None
                        else lambda x: preprocessor(x)
                    )
                    self._preprocessors.append(transform)
                    data = transform(data)
        return data

    def transform(self, data):
        if self._preprocessors is None:
            raise PreprocessorNotFittedError()
        for preprocessor in self._preprocessors:
            if isinstance(preprocessor, list):
                output = PreprocssingOutputDict(data)
                for index, encoder, ouput_name in preprocessor:
                    if isinstance(index, str) or (
                        isinstance(index, list) and isinstance(index[0], str)
                    ):
                        selection = data[index]
                    else:
                        try:
                            selection = data[:, index]
                        except:
                            selection = data[index]
                    output[output_name] = encoder(selection)
                data = output
            else:
                data = preprocessor(data)
        return data
