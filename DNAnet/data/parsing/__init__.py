def parse_called_alleles(*args, **kwargs):
    from DNAnet.data.parsing.parse_annotations import parse_called_alleles as _parse_called_alleles

    return _parse_called_alleles(*args, **kwargs)


def get_peak_data(*args, **kwargs):
    from DNAnet.data.parsing.parse_raw_hid import get_peak_data as _get_peak_data

    return _get_peak_data(*args, **kwargs)


__all__ = ["parse_called_alleles", "get_peak_data"]
