import { AbilityConstant, ConfigurationConstant, UIAbility, Want } from '@kit.AbilityKit';
import { hilog } from '@kit.PerformanceAnalysisKit';
import { window, AppStorageV2 } from '@kit.ArkUI';
import { ActiveFormState, APP_STORAGE_KEY_ACTIVE_FORM, ROUTER_KEY_FORM_ID } from '../common/ActiveFormatState'



const DOMAIN = 0x0000;
const TAG: string = 'HarmonyStock';



export default class EntryAbility extends UIAbility {
  // Ability 创建时调用（冷启动 / 重新拉起都会走这里）
  onCreate(want: Want, launchParam: AbilityConstant.LaunchParam): void {
    try {
      // 强制不跟随系统深浅色模式，交给应用自己控制主题
      this.context
        .getApplicationContext()
        .setColorMode(ConfigurationConstant.ColorMode.COLOR_MODE_NOT_SET)
    } catch (err) {
      // 设置失败不影响主流程，打日志即可
      hilog.error(
        DOMAIN,
        TAG,
        'Failed to set colorMode. Cause: %{public}s',
        JSON.stringify(err)
      )
    }

    hilog.info(DOMAIN, TAG, '%{public}s', 'Ability onCreate')

    // 冷启动场景：
    // 如果是从桌面卡片（Form）点击进入，会在 want 里带 formId
    // 这里统一取出来，写入 AppStorageV2，供页面层消费
    this.consumeFormIdIntoGlobalState(want, 'onCreate')
  }


  onNewWant(want: Want, launchParam: AbilityConstant.LaunchParam): void {
    hilog.info(DOMAIN, TAG, '%{public}s', 'Ability onNewWant');
    this.consumeFormIdIntoGlobalState(want, 'onNewWant');
  }

  onDestroy(): void {
    hilog.info(DOMAIN, TAG, '%{public}s', 'Ability onDestroy');
  }

  onWindowStageCreate(windowStage: window.WindowStage): void {
    hilog.info(DOMAIN, TAG, '%{public}s', 'Ability onWindowStageCreate');

    windowStage.loadContent('pages/Index', (err) => {
      if (err.code) {
        hilog.error(DOMAIN, TAG, 'Failed to load the content. Cause: %{public}s', JSON.stringify(err));
        return;
      }
      hilog.info(DOMAIN, TAG, 'Succeeded in loading the content.');
    });
  }

  onWindowStageDestroy(): void {
    hilog.info(DOMAIN, TAG, '%{public}s', 'Ability onWindowStageDestroy');
  }

  onForeground(): void {
    hilog.info(DOMAIN, TAG, '%{public}s', 'Ability onForeground');
  }

  onBackground(): void {
    hilog.info(DOMAIN, TAG, '%{public}s', 'Ability onBackground');
  }

  // 把桌面卡片（Form）通过 router/want 带进来的 formId 存到 AppStorageV2，给页面侧 onPageShow 去“消费”
  private consumeFormIdIntoGlobalState(want: Want, from: string): void {
    // 从 want 里读出 formId（桌面卡片跳转时由 router 传入）
    const formId: string = this.readStringParamFromWant(want, ROUTER_KEY_FORM_ID)
    if (formId.length === 0) {
      // 没有 formId：说明这次不是从桌面卡片进来的，直接忽略
      return
    }

    // 拿到全局状态对象（没有就按默认构造创建），用于跨 Ability / Page 传递 formId
    const st: ActiveFormState | undefined = AppStorageV2.connect(
      ActiveFormState,
      APP_STORAGE_KEY_ACTIVE_FORM,
      () => new ActiveFormState()
    )

    if (st !== undefined) {
      // 写入全局：页面侧会在 onPageShow 里读出来，然后清空，避免重复触发
      st.formId = formId
      hilog.info(DOMAIN, TAG, '[%{public}s] saved formId=%{public}s into AppStorageV2', from, formId)
    } else {
      // connect 失败一般是 key 未注册 / 上下文异常，打日志方便定位
      hilog.error(DOMAIN, TAG, 'AppStorageV2.connect failed')
    }
  }


  // 从 want.parameters 里安全读取一个字符串参数（不存在 / 非 string 都返回空串）
  private readStringParamFromWant(want: Want, key: string): string {
    // 没有 parameters，直接返回空
    if (!want.parameters) {
      return '';
    }

    // parameters 在 ArkTS 里是一个弱类型对象，这里统一当成 Record 来读
    const params: Record<string, Object> = want.parameters as Record<string, Object>;
    const v: Object | undefined = params[key];

    // key 不存在或值为空
    if (v === undefined || v === null) {
      return '';
    }

    // 只接受 string，其它类型一律丢弃，避免类型污染
    if (typeof v === 'string') {
      return (v as string).trim();
    }

    return '';
  }
}
