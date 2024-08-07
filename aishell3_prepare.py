"""
Data preparation.

Download: http://www.openslr.org/12

Author
------
 * Mirco Ravanelli, 2020
 * Ju-Chieh Chou, 2020
 * Loren Lugosch, 2020
 * Pierre Champion, 2023
 * Adel Moumen, 2024
"""

import csv
import functools
import logging
import os
import random
from collections import Counter
from dataclasses import dataclass

from speechbrain.dataio.dataio import (
    load_pkl,
    merge_csvs,
    read_audio_info,
    save_pkl,
)
from speechbrain.utils.data_utils import download_file, get_all_files
from speechbrain.utils.parallel import parallel_map

import zhconv

logger = logging.getLogger(__name__)
OPT_FILE = "opt_mandarin_prepare.pkl"
SAMPLERATE = 16000
OPEN_SLR_11_LINK = "http://www.openslr.org/resources/11/"
OPEN_SLR_11_NGRAM_MODELs = [
    "3-gram.arpa.gz",
    "3-gram.pruned.1e-7.arpa.gz",
    "3-gram.pruned.3e-7.arpa.gz",
    "4-gram.arpa.gz",
]


def prepare_aishell3(
    data_folder,
    save_folder,
    tr_splits=[],
    dev_splits=[],
    te_splits=[],
    select_n_sentences=None,
    merge_lst_tr=[],
    merge_name_tr=None,
    merge_lst_cv=[],
    merge_name_cv=None,
    merge_lst_tt=[],
    merge_name_tt=None,
    create_lexicon=False,
    skip_prep=False,
):
    """
    This class prepares the csv files for the mandarin dataset.
    Download link: http://www.openslr.org/12

    Arguments
    ---------
    data_folder : str
        Path to the folder where the original LibriSpeech dataset is stored.
    save_folder : str
        The directory where to store the csv files.
    tr_splits : list
        List of train splits to prepare from ['test-others','train-clean-100',
        'train-clean-360','train-other-500'].
    dev_splits : list
        List of dev splits to prepare from ['dev-clean','dev-others'].
    te_splits : list
        List of test splits to prepare from ['test-clean','test-others'].
    select_n_sentences : int
        Default : None
        If not None, only pick this many sentences.
    merge_lst : list
        List of librispeech splits (e.g, train-clean, train-clean-360,..) to
        merge in a single csv file.
    merge_name: str
        Name of the merged csv file.
    create_lexicon: bool
        If True, it outputs csv files containing mapping between grapheme
        to phonemes. Use it for training a G2P system.
    skip_prep: bool
        If True, data preparation is skipped.

    Returns
    -------
    None

    Example
    -------
    >>> data_folder = 'datasets/LibriSpeech'
    >>> tr_splits = ['train-clean-100']
    >>> dev_splits = ['dev-clean']
    >>> te_splits = ['test-clean']
    >>> save_folder = 'librispeech_prepared'
    >>> prepare_librispeech(data_folder, save_folder, tr_splits, dev_splits, te_splits)
    """

    if skip_prep:
        return
    data_folder = data_folder
    train_splits = tr_splits 
    test_splits = dev_splits + te_splits
    save_folder = save_folder
    select_n_sentences = select_n_sentences
    conf = {
        "select_n_sentences": select_n_sentences,
    }

    # Other variables
    # Saving folder
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    save_opt = os.path.join(save_folder, OPT_FILE)

    # Check if this phase is already done (if so, skip it)
    if skip(train_splits, save_folder, conf) and skip(test_splits, save_folder, conf):
        logger.info("Skipping preparation, completed in previous run.")
        return
    else:
        logger.info("Data_preparation...")

    # 這個不用的話應該可以註解掉，我只有檢查train set，沒有檢查test set
    # Additional checks to make sure the data folder contains aishell3
    check_aishell3_folders(data_folder, train_splits)

    # 處理 train_splits
    all_texts_train = process_splits(train_splits, "train", data_folder, save_folder, select_n_sentences)

    # 處理 test_splits
    all_texts_test = process_splits(test_splits, "test", data_folder, save_folder, select_n_sentences)

    # 這目前用不到，跳過
    # 更新所有文本字典
    # all_texts = {**all_texts_train, **all_texts_test}

    # Merging train.csv file if needed
    merge_csv_files(merge_lst_tr, merge_name_tr, "train", save_folder)

    # Merging valid.csv file if needed
    merge_csv_files(merge_lst_cv, merge_name_cv, "test", save_folder)

    # Merging test.csv file if needed
    merge_csv_files(merge_lst_tt, merge_name_tt, "test", save_folder)

    # 預設不會走這，先跳過
    # # Create lexicon.csv and oov.csv
    # if create_lexicon:
    #     create_lexicon_and_oov_csv(all_texts, save_folder)

    # saving options
    save_pkl(conf, save_opt)


def create_lexicon_and_oov_csv(all_texts, save_folder):
    """
    Creates lexicon csv files useful for training and testing a
    grapheme-to-phoneme (G2P) model.

    Arguments
    ---------
    all_texts : dict
        Dictionary containing text from the librispeech transcriptions
    save_folder : str
        The directory where to store the csv files.
    """
    # If the lexicon file does not exist, download it
    lexicon_url = "http://www.openslr.org/resources/11/librispeech-lexicon.txt"
    lexicon_path = os.path.join(save_folder, "librispeech-lexicon.txt")

    if not os.path.isfile(lexicon_path):
        logger.info(
            "Lexicon file not found. Downloading from %s." % lexicon_url
        )
        download_file(lexicon_url, lexicon_path)

    # Get list of all words in the transcripts
    transcript_words = Counter()
    for key in all_texts:
        transcript_words.update(all_texts[key].split("_"))

    # Get list of all words in the lexicon
    lexicon_words = []
    lexicon_pronunciations = []
    with open(lexicon_path, "r") as f:
        lines = f.readlines()
        for line in lines:
            word = line.split()[0]
            pronunciation = line.split()[1:]
            lexicon_words.append(word)
            lexicon_pronunciations.append(pronunciation)

    # Create lexicon.csv
    header = "ID,duration,char,phn\n"
    lexicon_csv_path = os.path.join(save_folder, "lexicon.csv")
    with open(lexicon_csv_path, "w") as f:
        f.write(header)
        for idx in range(len(lexicon_words)):
            separated_graphemes = [c for c in lexicon_words[idx]]
            duration = len(separated_graphemes)
            graphemes = " ".join(separated_graphemes)
            pronunciation_no_numbers = [
                p.strip("0123456789") for p in lexicon_pronunciations[idx]
            ]
            phonemes = " ".join(pronunciation_no_numbers)
            line = (
                ",".join([str(idx), str(duration), graphemes, phonemes]) + "\n"
            )
            f.write(line)
    logger.info("Lexicon written to %s." % lexicon_csv_path)

    # Split lexicon.csv in train, validation, and test splits
    split_lexicon(save_folder, [98, 1, 1])


def split_lexicon(data_folder, split_ratio):
    """
    Splits the lexicon.csv file into train, validation, and test csv files

    Arguments
    ---------
    data_folder : str
        Path to the folder containing the lexicon.csv file to split.
    split_ratio : list
        List containing the training, validation, and test split ratio. Set it
        to [80, 10, 10] for having 80% of material for training, 10% for valid,
        and 10 for test.
    """
    # Reading lexicon.csv
    lexicon_csv_path = os.path.join(data_folder, "lexicon.csv")
    with open(lexicon_csv_path, "r") as f:
        lexicon_lines = f.readlines()
    # Remove header
    lexicon_lines = lexicon_lines[1:]

    # Shuffle entries
    random.shuffle(lexicon_lines)

    # Selecting lines
    header = "ID,duration,char,phn\n"

    tr_snts = int(0.01 * split_ratio[0] * len(lexicon_lines))
    train_lines = [header] + lexicon_lines[0:tr_snts]
    valid_snts = int(0.01 * split_ratio[1] * len(lexicon_lines))
    valid_lines = [header] + lexicon_lines[tr_snts : tr_snts + valid_snts]
    test_lines = [header] + lexicon_lines[tr_snts + valid_snts :]

    # Saving files
    with open(os.path.join(data_folder, "lexicon_tr.csv"), "w") as f:
        f.writelines(train_lines)
    with open(os.path.join(data_folder, "lexicon_dev.csv"), "w") as f:
        f.writelines(valid_lines)
    with open(os.path.join(data_folder, "lexicon_test.csv"), "w") as f:
        f.writelines(test_lines)


@dataclass
class LSRow:
    snt_id: str
    spk_id: str
    duration: float
    file_path: str
    words: str


def process_line(wav_file, text_dict) -> LSRow:
    # 從路徑中提取檔案名稱
    file_name = wav_file.split("/")[-1]
    # 去掉 .wav 附檔名
    base_name = file_name.replace(".wav", "")
    # 提取 snt_id, 根據例子 snt_id 是 _ 之前的部分
    snt_id = base_name.split("_")[0]
    spk_id = snt_id[:7]
    
    if snt_id not in text_dict:
        raise KeyError(f"Missing key: {snt_id} in text_dict")
    
    wrds = text_dict[snt_id]
    wrds = " ".join(wrds.split("_"))

    info = read_audio_info(wav_file)
    duration = info.num_frames / info.sample_rate

    return LSRow(
        snt_id=snt_id,
        spk_id=spk_id,
        duration=duration,
        file_path=wav_file,
        words=wrds,
    )


def create_csv(save_folder, data_type, wav_lst, text_dict, split, select_n_sentences):
    """
    Create the dataset csv file given a list of wav files.

    Arguments
    ---------
    save_folder : str
        Location of the folder for storing the csv.
    wav_lst : list
        The list of wav files of a given data split.
    text_dict : list
        The dictionary containing the text of each sentence.
    split : str
        The name of the current data split.
    select_n_sentences : int, optional
        The number of sentences to select.

    Returns
    -------
    None
    """
    # Setting path for the csv file
    csv_file = os.path.join(save_folder, f"{data_type}_{split}.csv")
    if os.path.exists(csv_file):
        logger.info("Csv file %s already exists, not recreating." % csv_file)
        return

    # Preliminary prints
    msg = "Creating csv lists in  %s..." % (csv_file)
    logger.info(msg)

    csv_lines = [["ID", "duration", "wav", "spk_id", "wrd"]]
    snt_cnt = 0
    line_processor = functools.partial(process_line, text_dict=text_dict)
    # Processing all the wav files in wav_lst
    # FLAC metadata reading is already fast, so we set a high chunk size
    # to limit main thread CPU bottlenecks
    for row in parallel_map(line_processor, wav_lst, chunk_size=8192):
        csv_line = [
            row.snt_id,
            str(row.duration),
            row.file_path,
            row.spk_id,
            row.words,
        ]

        # Appending current file to the csv_lines list
        csv_lines.append(csv_line)

        snt_cnt = snt_cnt + 1

        # parallel_map guarantees element ordering so we're OK
        if snt_cnt == select_n_sentences:
            break

    # Writing the csv_lines
    with open(csv_file, mode="w") as csv_f:
        csv_writer = csv.writer(
            csv_f, delimiter=",", quotechar='"', quoting=csv.QUOTE_MINIMAL
        )

        for line in csv_lines:
            csv_writer.writerow(line)

    # Final print
    msg = "%s successfully created!" % (csv_file)
    logger.info(msg)


def skip(splits, save_folder, conf):
    """
    Detect when the mandarin data prep can be skipped.

    Arguments
    ---------
    splits : list
        A list of the splits expected in the preparation.
    save_folder : str
        The location of the save directory
    conf : dict
        The configuration options to ensure they haven't changed.

    Returns
    -------
    bool
        if True, the preparation phase can be skipped.
        if False, it must be done.
    """

    # Checking csv files
    skip = True

    for split in splits:
        if not os.path.isfile(os.path.join(save_folder, split + ".csv")):
            skip = False

    #  Checking saved options
    save_opt = os.path.join(save_folder, OPT_FILE)
    if skip is True:
        if os.path.isfile(save_opt):
            opts_old = load_pkl(save_opt)
            if opts_old == conf:
                skip = True
            else:
                skip = False
        else:
            skip = False

    return skip

# 修改以符合aishell3 content.txt處理方式
def text_to_dict(text_lst):
    """
    This converts lines of text into a dictionary and converts Simplified Chinese to Traditional Chinese.

    Arguments
    ---------
    text_lst : list of str
        List of paths to the files containing the Mandarin text transcription.

    Returns
    -------
    dict
        The dictionary containing the text transcriptions for each sentence.
    """
    # 初始化文本字典
    text_dict = {}

    # 讀取所有的轉錄文件
    for file in text_lst:
        with open(file, "r", encoding="utf-8") as f:
            # 讀取轉錄文件中的每一行
            for line in f:
                line_lst = line.strip().split("\t")
                filename_with_ext = line_lst[0]
                # 去掉副檔名
                filename = os.path.splitext(filename_with_ext)[0]
                # 刪除拼音部分，保留漢字部分
                chinese_text = " ".join(word for word in line_lst[1].split(" ") if not any(char.isdigit() for char in word))
                # 將簡體中文轉換成繁體中文
                traditional_text = zhconv.convert(chinese_text, 'zh-tw')
                text_dict[filename] = traditional_text.replace(" ", "_")  # 使用下劃線連接漢字

    return text_dict

def check_aishell3_folders(data_folder, splits):
    """
    Check if the data folder actually contains the mandarin dataset.

    If it does not, an error is raised.

    Arguments
    ---------
    data_folder : str
        The path to the directory with the data.
    splits : list
        The portions of the data to check.

    Raises
    ------
    OSError
        If aishell3 is not found at the specified path.
    """
    # Checking if all the splits exist
    for split in splits:
        split_folder = os.path.join(data_folder, "train/wav", split)
        if not os.path.exists(split_folder):
            err_msg = (
                "the folder %s does not exist (it is expected in the "
                "mandarin dataset)" % split_folder
            )
            raise OSError(err_msg)


def download_librispeech_vocab_text(destination):
    """Download librispeech vocab file and unpack it.

    Arguments
    ---------
    destination : str
        Place to put vocab file.
    """
    f = "librispeech-vocab.txt"
    download_file(OPEN_SLR_11_LINK + f, destination)


def download_openslr_librispeech_lm(destination, rescoring_lm=True):
    """Download openslr librispeech lm and unpack it.

    Arguments
    ---------
    destination : str
        Place to put lm.
    rescoring_lm : bool
        Also download bigger 4grams model
    """
    os.makedirs(destination, exist_ok=True)
    for f in OPEN_SLR_11_NGRAM_MODELs:
        if f.startswith("4") and not rescoring_lm:
            continue
        d = os.path.join(destination, f)
        download_file(OPEN_SLR_11_LINK + f, d, unpack=True)


def download_sb_librispeech_lm(destination, rescoring_lm=True):
    """Download sb librispeech lm and unpack it.

    Arguments
    ---------
    destination : str
        Place to put lm.
    rescoring_lm : bool
        Also download bigger 4grams model
    """
    os.makedirs(destination, exist_ok=True)
    download_file(
        "https://www.dropbox.com/scl/fi/3fkkdlliavhveb5n3nsow/3gram_lm.arpa?rlkey=jgdrluppfut1pjminf3l3y106&dl=1",
        os.path.join(destination, "3-gram_sb.arpa"),
    )
    if rescoring_lm:
        download_file(
            "https://www.dropbox.com/scl/fi/roz46ee0ah2lvy5csno4z/4gram_lm.arpa?rlkey=2wt8ozb1mqgde9h9n9rp2yppz&dl=1",
            os.path.join(destination, "4-gram_sb.arpa"),
        )

def process_splits(splits, data_type, data_folder, save_folder, select_n_sentences=None):
    all_texts = {}
    
    for split_index in range(len(splits)):
        split = splits[split_index]

        wav_lst = get_all_files(
            os.path.join(data_folder, data_type, "wav", split), match_and=[".wav"]
        )
        
        text_lst = get_all_files(
            os.path.join(data_folder, data_type), match_and=["content.txt"]
        )
        
        text_dict = text_to_dict(text_lst)
        all_texts.update(text_dict)

        # 這裡指定select_n_sentences可能會報錯，不過我暫時沒有要挑句子，所以先跳過
        if select_n_sentences is not None:
            n_sentences = select_n_sentences[split_index]
        else:
            n_sentences = len(wav_lst)
        
        create_csv(save_folder, data_type, wav_lst, text_dict, split, n_sentences)
    
    return all_texts

def merge_csv_files(merge_list, merge_name, file_type, save_folder):
    if merge_list and merge_name is not None:
        merge_files = [f"{file_type}_{split}.csv" for split in merge_list]
        merge_csvs(data_folder=save_folder, csv_lst=merge_files, merged_csv=merge_name)