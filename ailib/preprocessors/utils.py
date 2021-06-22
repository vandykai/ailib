"""Wrapper function organizes a number of transform functions."""
import typing
import functools

from .units import Unit, StatefulUnit, Vocabulary
from ailib.data_pack import DataPack
from tqdm.auto import tqdm

def chain_transform(units: typing.List[Unit]) -> typing.Callable:
    """
    Compose unit transformations into a single function.

    :param units: List of :class:`matchzoo.StatelessUnit`.
    """

    @functools.wraps(chain_transform)
    def wrapper(arg):
        """Wrapper function of transformations composition."""
        for unit in units:
            arg = unit.transform(arg)
        return arg

    unit_names = ' => '.join(unit.__class__.__name__ for unit in units)
    wrapper.__name__ += ' of ' + unit_names
    return wrapper


def build_unit_from_data_pack(
    unit: StatefulUnit,
    data_pack: DataPack, mode: str = 'both',
    flatten: bool = True, verbose: int = 1
) -> StatefulUnit:
    """
    Build a :class:`StatefulUnit` from a :class:`DataPack` object.

    :param unit: :class:`StatefulUnit` object to be built.
    :param data_pack: The input :class:`DataPack` object.
    :param mode: One of 'left', 'right', and 'both', to determine the source
            data for building the :class:`VocabularyUnit`.
    :param flatten: Flatten the datapack or not. `True` to organize the
        :class:`DataPack` text as a list, and `False` to organize
        :class:`DataPack` text as a list of list.
    :param verbose: Verbosity.
    :return: A built :class:`StatefulUnit` object.

    """
    corpus = []
    if flatten:
        data_pack.apply_on_text(corpus.extend, mode=mode, verbose=verbose)
    else:
        data_pack.apply_on_text(corpus.append, mode=mode, verbose=verbose)
    if verbose:
        description = 'Building ' + unit.__class__.__name__ + \
                      ' from a datapack.'
        corpus = tqdm(corpus, desc=description)
    unit.fit(corpus)
    return unit

def build_vocab_unit(
    data_pack: DataPack,
    mode: str = 'both',
    verbose: int = 1
) -> Vocabulary:
    """
    Build a :class:`preprocessor.units.Vocabulary` given `data_pack`.

    The `data_pack` should be preprocessed forehand, and each item in
    `text_left` and `text_right` columns of the `data_pack` should be a list
    of tokens.

    :param data_pack: The :class:`DataPack` to build vocabulary upon.
    :param mode: One of 'left', 'right', and 'both', to determine the source
    data for building the :class:`VocabularyUnit`.
    :param verbose: Verbosity.
    :return: A built vocabulary unit.

    """
    return build_unit_from_data_pack(
        unit=Vocabulary(),
        data_pack=data_pack,
        mode=mode,
        flatten=True, verbose=verbose
    )
